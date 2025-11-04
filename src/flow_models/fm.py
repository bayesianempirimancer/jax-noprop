import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
from flax.core import FrozenDict
import optax
from typing import Tuple, Dict, Optional
from dataclasses import dataclass, field

from functools import partial, cached_property

# Import directly without going through src package to avoid einops dependency
from src.configs.base_config import BaseConfig
from src.utils.ode_integration import integrate_ode
from src.models.vae.encoders import create_encoder
from src.models.vae.decoders import create_decoder
from src.flow_models.crn import create_conditional_resnet
from src.embeddings.noise_schedules import (
    create_noise_schedule,
)


@dataclass(frozen=True)
class VAEFlowConfig(BaseConfig):
    """Configuration for VAE with flow model using separate dictionaries."""
    # BaseConfig fields
    model_name: str = "vae_flow_network"

    main: FrozenDict = field(default_factory=lambda: FrozenDict({
        "input_shape": "NA",  # Will be set based on z_dim
        "output_shape": "NA",  # Will be set based on z_dim or z_dim**2
        "latent_shape": "NA",  # Will be set based on x_dim
        "recon_loss_type": "cross_entropy", # Options: "cross_entropy", "mse", "none".  Should be consistent with decoder type
        "recon_weight": 0.1,  # Weight for reconstruction loss in total loss
        "reg_weight": 0.0,  # Weight for regularization loss in total loss
        "sigma": 0.02,  # Noise level for flow matching
        "integration_method": "euler",  # Options: "euler", "heun", "rk4", "adaptive", "midpoint"
                                           # Only use midpoint for diffusion model.  singularities at t=0 break forward integration.
        "encode_x": False,  # Whether to encode x before passing to CRN (True for sequences, False for backward compatibility)
    }))
    
    noise_schedule: FrozenDict = field(default_factory=lambda: FrozenDict({
        "schedule_type": "exponential",  # Type of schedule (linear, exponential, cosine, sigmoid, cauchy, laplace, logistic, quadratic, polynomial, monotonic_nn, learnable, network)
        "learnable": True,  # Whether schedule parameters are learnable (False uses stop_gradient)
        "hidden_dims": (64, 64),  # Hidden dimensions for NoiseScheduleNetwork schedule
        # Comprehensive default parameters for all schedules (common naming convention)
        "default_params": FrozenDict({
            # Linear, Exponential, Cauchy, Laplace schedules
            "alpha_bar_min": 0.01,
            "alpha_bar_max": 0.99,
            # Cosine schedule
            "s": 0.008,
            # Sigmoid, Logistic schedules
            "k": 10.0,
            "t_mid": 0.5,
            # Exponential schedule
            "beta": 2.0,
            # Cauchy, Laplace schedules
            "loc": 0.5,
            "scale": 0.1,
            # Polynomial schedule
            "power": 2.0,
            # NoiseScheduleNetwork schedule
            "gamma_range": (-4.0, 4.0),
        }),
    }))

    crn: FrozenDict = field(default_factory=lambda: FrozenDict({
        "model_type": "vanilla", # Options: "vanilla", "geometric", "potential", "natural"
        "network_type": "mlp", # Options: "mlp", "bilinear", "convex"
        "hidden_dims": (32, 32, 32, 32, 32),
        "time_embed_dim": 64,
        "time_embed_method": "sinusoidal",
        "activation_fn": "swish",
        "use_batch_norm": False,
        "dropout_rate": 0.1,
    }))
    
    encoder: FrozenDict = field(default_factory=lambda: FrozenDict({
        "model_type": "identity", # Options: "mlp", "mlp_normal", "resnet", "resnet_normal", "identity", "linear"
        "encoder_type": "deterministic",  # Options: "deterministic", "normal", 
        "input_shape": "NA",  # Will be set from main config if not specified
        "latent_shape": "NA",
        "hidden_dims": (16,16),
        "activation": "swish",
        "dropout_rate": 0.1,
    }))
    
    decoder: FrozenDict = field(default_factory=lambda: FrozenDict({
        "model_type": "identity", # Options: "mlp", "resnet", "identity", 'linear'
        "decoder_type": "none", # Options: "linear", "softmax", "none"
        "latent_shape": "NA",  # Will be set from main config if not specified
        "output_shape": "NA",
        "hidden_dims": (32, 16),
        "activation": "swish",
        "dropout_rate": 0.1,
    }))


class VAE_flow(nn.Module):
    """Variational Autoencoder with flow model using @nn.compact methods."""
    config: VAEFlowConfig
    
    def setup(self):
        """Initialize the CRN model and noise schedule as fields."""
        # For generative mode, we need to handle the case where x=None
        # We'll create the CRN model with a proper input shape that can handle None
        input_shape = self.config.main["input_shape"]
        
        # If input_shape is (1,) (dummy for generative mode), we need to handle this differently
        # The CRN model will be called with x=None, so we need to ensure it can handle this
        self.crn_model = create_conditional_resnet(
            self.config.crn,
            latent_shape=self.z_shape,
            input_shape=input_shape,
            output_shape=self.z_shape
        )
        
        # Initialize noise schedule using factory function
        # Get schedule type from noise_schedule config or fallback to main config
        schedule_config = self.config.noise_schedule if hasattr(self.config, 'noise_schedule') else FrozenDict()
        schedule_type = schedule_config.get("schedule_type", self.config.main.get("noise_schedule", "linear"))
        
        # Store whether schedule parameters should be learnable
        self.noise_schedule_learnable = schedule_config.get("learnable", True)
        
        # Create schedule using factory - pass learnable flag to schedule
        # The schedule will handle stop_gradient internally if learnable=False
        # Schedule classes use their own defaults for parameters
        self.noise_schedule = create_noise_schedule(
            schedule_type, 
            learnable=self.noise_schedule_learnable
        )
        
        # Initialize encoder and decoder
        # Use shapes directly from encoder/decoder configs
        # For sequences, these should be feature-only shapes (e.g., (2,) instead of (48, 2))
        self.encoder = create_encoder(
            self.config.encoder,
            input_shape=self.config.encoder["input_shape"],
            latent_shape=self.config.encoder["latent_shape"]
        )
        
        # Decoder maps from latent to output
        self.decoder = create_decoder(
            self.config.decoder,
            latent_shape=self.config.decoder["latent_shape"],
            output_shape=self.config.decoder["output_shape"]
        )
    

    @property
    def z_shape(self) -> Tuple[int, ...]:
        """Effective z_shape from config."""
        return self.config.main["latent_shape"]
    
    @property
    def z_ndims(self) -> int:
        """Number of dimensions in z_shape."""
        return len(self.z_shape)
    
    @property
    def y_ndims(self) -> int:
        """Number of dimensions in y_shape."""
        return len(self.config.main["output_shape"])
    
    @cached_property
    def z_dim(self) -> int:
        """Total flattened dimension of z."""
        z_dim = 1
        for dim in self.z_shape:
            z_dim *= dim
        return z_dim

    @property
    def x_ndims(self) -> int:
        """Number of dimensions in x_shape."""
        return len(self.config.main["input_shape"])

    def _flatten_z(self, z: jnp.ndarray) -> jnp.ndarray:
        """Flatten the z tensor to 1D for processing."""
        if len(self.z_shape) <= 1:
            return z
        return z.reshape(z.shape[:-self.z_ndims] + (self.z_dim,))

    def _unflatten_z(self, z: jnp.ndarray) -> jnp.ndarray:
        """Unflatten the z tensor back to original shape."""
        if len(self.z_shape) <= 1:
            return z
        return z.reshape(z.shape[:-1] + self.z_shape)
    
    def _broadcast_inputs(self, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        z_batch_shape = z.shape[:-self.z_ndims]
        x_batch_shape = () if x is None else x.shape[:-self.x_ndims]
        t_batch_shape = () if t is None else jnp.asarray(t).shape
        batch_shape = jnp.broadcast_shapes(z_batch_shape, x_batch_shape, t_batch_shape)

        z = jnp.broadcast_to(z, batch_shape + self.z_shape)
        x = None if x is None else jnp.broadcast_to(x, batch_shape + x.shape[-self.x_ndims:])
        t = None if t is None else jnp.broadcast_to(t, batch_shape)
        return z, x, t

    @nn.compact
    def dz_dt(self, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """Flow model that computes dz/dt using CRN."""
        # Optimized: minimize broadcasting and avoid redundant operations
        t = jnp.asarray(t)
        z, x, t = self._broadcast_inputs(z, x, t)
        
        # For seq to seq models we must encode x in the same manner as y before passing to seq_crn
        if self.config.main.get("encode_x", False) and x is not None:
            x = self.encode(x, training)[0]
        
        dz_dt = self.crn_model(z, x, t, training=training)            
        return dz_dt

    def lazy_flow(self, z_t, z_target, alpha_t, gamma_prime_t):
        return gamma_prime_t * (jnp.sqrt(alpha_t)* z_target - (0.5*(1.0 + alpha_t)) * z_t)

    # KL divergence with SNR weighting: SNR_noise_weight = gamma_prime
    # Note:  0.5 is dropped and the proof is here: https://arxiv.org/html/2312.10393v1
    def lazy_noise_snr(self, alpha_t, gamma_prime_t):  return gamma_prime_t
    def lazy_target_snr(self, alpha_t, gamma_prime_t): return gamma_prime_t * alpha_t / (1.0 - alpha_t)
    def lazy_error_snr(self, alpha_t, gamma_prime_t):  return gamma_prime_t / (1.0 - alpha_t)
    def lazy_score_snr(self, alpha_t, gamma_prime_t):  return gamma_prime_t * (1.0 - alpha_t)
    def lazy_flow_snr(self, alpha_t, gamma_prime_t):   return  1.0 / ((1.0 - alpha_t) * alpha_t * gamma_prime_t)

    def lazy_score(self, z_t, z_target, alpha_t):     return (jnp.sqrt(alpha_t)*z_target - z_t) / (1.0 - alpha_t)
    def lazy_error(self, z_t, z_target, alpha_t):     return z_t - jnp.sqrt(alpha_t)* z_target
    def lazy_noise(self, z_t, z_target, alpha_t):     return self.lazy_error(z_t, z_target, alpha_t) / jnp.sqrt(1.0 - alpha_t)

    def lazy_KL_from_target(self, z_target_est, z_target, alpha_t, gamma_prime_t):
        diff = z_target - z_target_est
        return self.lazy_target_snr(alpha_t, gamma_prime_t) * jnp.sum(diff ** 2, axis=tuple(range(-self.z_ndims, 0)))

    def lazy_KL_from_noise(self, noise_est, z_t, z_target, alpha_t, gamma_prime_t):
        noise_diff = self.lazy_noise(z_t, z_target, alpha_t) - noise_est
        return self.lazy_noise_snr(alpha_t, gamma_prime_t) * jnp.sum(noise_diff ** 2, axis=tuple(range(-self.z_ndims, 0)))

    def lazy_KL_from_score(self, score_est, z_t, z_target, alpha_t, gamma_prime_t):
        score_diff = self.lazy_score(z_t, z_target, alpha_t) - score_est
        return self.lazy_score_snr(alpha_t, gamma_prime_t) * jnp.sum(score_diff ** 2, axis=tuple(range(-self.z_ndims, 0)))

    def lazy_KL_from_error(self, error_est, z_t, z_target, alpha_t, gamma_prime_t):
        error_diff = self.lazy_error(z_t, z_target, alpha_t) - error_est
        return self.lazy_error_snr(alpha_t, gamma_prime_t) * jnp.sum(error_diff ** 2, axis=tuple(range(-self.z_ndims, 0)))

    def lazy_KL_from_flow(self, flow_est, z_t, z_target, alpha_t, gamma_prime_t):
        noise = self.lazy_noise(z_t, z_target, alpha_t)
        flow_diff = self.lazy_flow(z_t, noise, alpha_t, gamma_prime_t) - flow_est
        return self.lazy_flow_snr * jnp.sum(flow_diff ** 2, axis=tuple(range(-self.z_ndims, 0))) / (gamma_prime_t ** 2)

    @nn.compact
    def get_noise_params(self, t: jnp.ndarray):
        """Get noise schedule output using @nn.compact method."""
        alpha_bar_t, gamma_prime_t = self.noise_schedule.get_alpha_bar_gamma_prime(t)
        return alpha_bar_t, gamma_prime_t
    
    
    @nn.compact
    def encode(self, x: jnp.ndarray, training: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Encoder that maps x to latent space."""
        encoder_output = self.encoder(x, training)
        if isinstance(encoder_output, tuple):
            return encoder_output   # Normal encoder returns (mu, logvar) tuple
        else:
            return encoder_output, -jnp.inf  # Deterministic encoder returns z, so (z, -jnp.inf) for consistency

    @nn.compact
    def decode(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """Decoder that maps latent z to output space."""
        return self.decoder(x, training)

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray, key: jr.PRNGKey, training: bool = True) -> jnp.ndarray:
        # For initialization, we need to call the nn compact methods to initialize parameters

        # Handle generative mode where x is None
        batch_shape = y.shape[:-self.y_ndims]
        # Call flow_model to initialize its parameters (need dummy z and t)
        dummy_z = jnp.zeros(batch_shape + self.z_shape)
        dummy_t = jnp.zeros(batch_shape)
        
        # Call get_noise_params to initialize noise schedule parameters
        self.get_noise_params(dummy_t)
        
        # initialize model components
        flow_output = self.dz_dt(dummy_z, x, dummy_t, training)
        encoder_output = self.encode(y, training)
        decoder_output = self.decode(dummy_z, training)
        
        return jnp.zeros(batch_shape + self.z_shape) # batch consistent with expectations

    @partial(jax.jit, static_argnums=(0, 5))    
    def loss(self, params: dict, x: Optional[jnp.ndarray], y: jnp.ndarray, key: jr.PRNGKey, training: bool = True) -> Tuple[jnp.ndarray, dict]:
        """
        Compute the loss by calling individual @nn.compact methods with proper rngs.
        """
        # Split keys for random sampling operations (not dropout)
        key, t_key, z_0_key, z_t_noise_key, z_target_key = jr.split(key, 5)
        # Keep key for dropout RNGs - will be automatically managed by @nn.compact
        batch_shape = y.shape[:-self.y_ndims]
        
        # Encode Target (noisy latent) - @nn.compact methods handle RNG automatically
        mu_z_target, logvar_z_target = self.apply(params, y, method='encode', training=training, rngs={'dropout': key})
        z_target = mu_z_target + jnp.exp(0.5 * logvar_z_target) * jr.normal(z_target_key, mu_z_target.shape)

        # Sample initial latent state and time
        z_0 = jr.normal(z_0_key, batch_shape + self.z_shape)
        t = jr.uniform(t_key, batch_shape + self.z_ndims*(1,), minval=0.0, maxval=1.0)
        squeezed_t = t.squeeze(tuple(range(-self.z_ndims, 0)))
        
        # Sample latent state at time t
        diff_z = z_target - z_0
        z_t = z_0 + t * diff_z 
        # z_t = jnp.sqrt(alpha_t) * z_target +  jnp.sqrt(1.0 - alpha_t) * z_0 
        # z_t = z_t + 2*(jnp.sqrt(alpha_t) - alpha_t) * jr.normal(z_t_noise_key, z_t.shape)

        # Compute Losses
        alpha_t, gamma_prime_t = self.apply(params, squeezed_t, method='get_noise_params')
        
        use_snr_weight = self.config.main.get("use_snr_weight", True)
        if use_snr_weight:
            snr_weight = self.lazy_flow_snr(alpha_t, gamma_prime_t)
        else: 
            snr_weight = 1.0
        snr_weight_mean = jnp.mean(snr_weight)
        
        dz_dt = self.apply(params, z_t, x, squeezed_t, method='dz_dt', training=training, rngs={'dropout': key})    
        z_target_est = dz_dt * (1.0-t) + z_t
        y_pred = self.apply(params, z_target_est, method='decode', training=training, rngs={'dropout': key})

        # Compute Lossess
        reg_loss = jnp.mean(dz_dt ** 2, axis=tuple(range(-self.z_ndims, 0)))
        reg_loss = jnp.mean(snr_weight * reg_loss)

#        dz_dt_target = self.lazy_flow(z_t, z_target, alpha_t, gamma_prime_t)
        flow_loss = jnp.mean((dz_dt - diff_z)**2, axis=tuple(range(-self.z_ndims, 0)))
        flow_loss = jnp.mean(snr_weight * flow_loss)

        reg_loss = jnp.mean(dz_dt**2, axis=tuple(range(-self.z_ndims, 0)))  # has batch_shape
        reg_loss = jnp.mean(snr_weight * reg_loss)

        recon_loss_type = self.config.main.get("recon_loss_type", "mse")
        if recon_loss_type == "cross_entropy":
            recon_loss = jnp.mean(-y * jnp.log(y_pred + 1e-8), axis = tuple(range(-self.y_ndims, 0)))
        elif recon_loss_type == "mse":
            recon_loss = jnp.mean((y - y_pred)**2, axis=tuple(range(-self.y_ndims, 0)))
        else:
            recon_loss = 0.0
        recon_loss = jnp.mean(self.lazy_target_snr(alpha_t, gamma_prime_t)*recon_loss)  # Average over batch dimension if needed        
   

        reg_weight = self.config.main.get("reg_weight", 0.0)
        recon_weight = self.config.main.get("recon_weight", 0.0)

        total_loss = flow_loss + recon_weight * recon_loss + reg_weight * reg_loss
        # total_loss = total_loss/snr_weight_mean

        return total_loss, {'flow_loss': flow_loss, 
                            'recon_loss': recon_loss, 
                            'reg_loss': reg_loss, 
                            'total_loss': total_loss}


    @partial(jax.jit, static_argnums=(0, 3, 4, 5))  # self, num_steps, integration_method, output_type, and training are static arguments
    def predict(self, params: dict, x: jnp.ndarray, num_steps: int = 20, integration_method: str = "euler", output_type: str = "end_point", prng_key: jr.PRNGKey = None) -> jnp.ndarray:
        """
        Make predictions using ODE solver integration.        
        Requires x is not None... use sample method for unconditional generation.
        """
        params_no_grad = jax.lax.stop_gradient(params)
        batch_shape = x.shape[:-self.x_ndims]
        
        # Generate flattened initial latent state z_0
        if prng_key is not None:
            z_0 = jr.normal(prng_key, batch_shape + (self.z_dim,))
        else:
            z_0 = jnp.zeros(batch_shape + (self.z_dim,))  # ode expects vectorized z
        
        def vector_field(params, z, x, t):
            z = self._unflatten_z(z)
            dz_dt = self.apply(params, z, x, t, method='dz_dt', training=False)
            return self._flatten_z(dz_dt)  # ODE solver expects flattened z
        
        # Integrate the ODE from t=0 to t=1
        z = integrate_ode(
            vector_field=vector_field,
            params=params_no_grad,
            z0=z_0,
            x=x,
            time_span=(0.0, 1.0),
            num_steps=num_steps,
            method=integration_method,
            output_type=output_type
        )
        z = self._unflatten_z(z)
        return self.apply(params, z, method='decode', training=False)
    
    @partial(jax.jit, static_argnums=(0, 3, 4, 5, 6))  # self, num_steps, integration_method, output_type are static arguments
    def sample(self, params: dict, prng_key: jr.PRNGKey, batch_shape: Tuple[int, ...], num_steps: int = 20, integration_method: str = "euler", output_type: str = "end_point") -> jnp.ndarray:
        """
        Generate samples using ODE solver for situations without conditional input (x=None).  
        Noise is injected at the initial timestep for the solver.
        Returns:
            If output_type="end_point": Final samples [batch_shape + y_shape]
            If output_type="trajectory": Full trajectory [num_steps, batch_shape + y_shape]
        """
        params_no_grad = jax.lax.stop_gradient(params)
        z_0 = jr.normal(prng_key, batch_shape + (self.z_dim,))
        # Define the vector field for ODE integration using flow_model method with x=None
        def vector_field(params, z, x, t):
            z = self._unflatten_z(z)
            dz_dt = self.apply(params, z, None, t, method='dz_dt', training=False)
            return self._flatten_z(dz_dt)
        
        # Integrate the ODE from t=0 to t=1
        z_trajectory = integrate_ode(
            vector_field=vector_field,
            params=params_no_grad,
            z0=z_0,
            x=None,  # No conditional input
            time_span=(0.0, 1.0),
            num_steps=num_steps,
            method=integration_method,
            output_type=output_type
        )
        return self.apply(params, z_trajectory, method='decode', training=False)
    

    @partial(jax.jit, static_argnums=(0, 5, 7))  # self, optimizer, and training are static arguments
    def train_step(self, params: dict, x: Optional[jnp.ndarray], y: jnp.ndarray, opt_state: dict, optimizer, key: jr.PRNGKey, training: bool = True) -> Tuple[dict, dict, jnp.ndarray, dict]:
        """
        JIT-compiled training step for VAE with flow model.
        
        Args:
            params: Current model parameters
            x: Conditional input [batch_size, input_dim]
            y: Target output [batch_size, output_dim]
            opt_state: Optimizer state
            optimizer: Optax optimizer
            key: Random key
            use_dropout: Whether to use dropout during training
            
        Returns:
            params: Updated model parameters
            opt_state: Updated optimizer state
            loss: Training loss
            metrics: Training metrics
        """
        # Compute loss and gradients
        def loss_fn(params):
            return self.loss(params, x, y, key, training=training)  
        
        (loss, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        
        # Update parameters using optimizer
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss, metrics
    
    @partial(jax.jit, static_argnums=(0, 5, 7))  # self and optimizer are static arguments
    def train_step_without_dropout(self, params: dict, x: jnp.ndarray, y: jnp.ndarray, opt_state: dict, optimizer, key: jr.PRNGKey, training: bool = False) -> Tuple[dict, dict, jnp.ndarray, dict]:
        # Compute loss and gradients
        def loss_fn(params):
            return self.loss(params, x, y, key, training=training)  
        
        (loss, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        
        # Update parameters using optimizer
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss, metrics