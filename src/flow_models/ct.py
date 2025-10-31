import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
from flax.core import FrozenDict
import optax
from typing import Tuple, Dict
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
    NoiseSchedule,
    LinearNoiseSchedule,
    CosineNoiseSchedule,
    SigmoidNoiseSchedule,
    ExponentialNoiseSchedule,
    CauchyNoiseSchedule,
    LaplaceNoiseSchedule,
    LogisticNoiseSchedule,
    NoiseScheduleNetwork,
    LearnableNoiseSchedule  # Alias for NoiseScheduleNetwork
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
        "integration_method": "midpoint",  # Options: "euler", "heun", "rk4", "adaptive", "midpoint"
                                           # Only use midpoint for diffusion model.  singularities at t=0 break forward integration.
        "noise_schedule": "linear",  # Options: "linear", "cosine", "sigmoid", "exponential", "cauchy", "laplace", "logistic", "quadratic", "polynomial", "monotonic_nn", "learnable", "network"
    }))
    
    noise_schedule: FrozenDict = field(default_factory=lambda: FrozenDict({
        "schedule_type": "linear",  # Type of schedule
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
        "hidden_dims": (64, 64, 64),
        "time_embed_dim": 64,
        "time_embed_method": "sinusoidal",
        "activation_fn": "swish",
        "use_batch_norm": False,
        "dropout_rate": 0.1,
    }))
    
    encoder: FrozenDict = field(default_factory=lambda: FrozenDict({
        "model_type": "linear", # Options: "mlp", "mlp_normal", "resnet", "resnet_normal", "identity", "linear"
        "encoder_type": "deterministic",  # Options: "deterministic", "normal", 
        "input_shape": "NA",  # Will be set from main config if not specified
        "latent_shape": "NA",
        "hidden_dims": (8,),
        "activation": "swish",
        "dropout_rate": 0.1,
    }))
    
    decoder: FrozenDict = field(default_factory=lambda: FrozenDict({
        "model_type": "identity", # Options: "mlp", "resnet", "identity"
        "decoder_type": "none", # Options: "linear", "softmax", "none"
        "latent_shape": "NA",  # Will be set from main config if not specified
        "output_shape": "NA",
        "hidden_dims": (64, 32, 16),
        "activation": "swish",
        "dropout_rate": 0.1,
    }))


class VAE_flow(nn.Module):
    """Variational Autoencoder with continuous-time flow model using @nn.compact methods.
    
    This class implements a VAE with a continuous-time flow model that uses sophisticated
    noise schedules and SNR-weighted loss functions. It combines the stability features
    from df.py with the advanced CT features from ct_orig.py.
    """
    config: VAEFlowConfig
    
    def setup(self):
        """Initialize the CRN model and noise schedule as fields."""
        self.crn_model = create_conditional_resnet(
            self.config.crn,
            latent_shape=self.z_shape,
            input_shape=self.config.main["input_shape"],
            output_shape=self.z_shape
        )
        
        # Initialize noise schedule using factory function
        # Get schedule type from noise_schedule config or fallback to main config
        schedule_config = self.config.noise_schedule if hasattr(self.config, 'noise_schedule') else FrozenDict()
        schedule_type = schedule_config.get("schedule_type", self.config.main.get("noise_schedule", "linear"))
        
        # Store whether schedule parameters should be learnable
        self.noise_schedule_learnable = schedule_config.get("learnable", True)
        
        # Extract initial parameter values from config (if provided)
        # Start with default_params, then override with any schedule-specific values
        default_params = schedule_config.get("default_params", FrozenDict())
        schedule_kwargs = dict(default_params)  # Start with defaults
        
        # Add hidden_dims from top level if present (for NoiseScheduleNetwork schedule)
        if "hidden_dims" in schedule_config:
            schedule_kwargs["hidden_dims"] = schedule_config["hidden_dims"]
        
        # Override with any schedule-specific parameters if provided
        # These take precedence over defaults
        param_names = [
            "alpha_bar_min", "alpha_bar_max", "s", "k", "t_mid",
            "beta", "loc", "scale", "power", "gamma_range"
        ]
        for param_name in param_names:
            if param_name in schedule_config:
                schedule_kwargs[param_name] = schedule_config[param_name]
        
        # Create schedule using factory - pass learnable flag to schedule
        # The schedule will handle stop_gradient internally if learnable=False
        self.noise_schedule = create_noise_schedule(
            schedule_type, 
            learnable=self.noise_schedule_learnable,
            **schedule_kwargs
        )
    
    def _broadcast_inputs(self, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        z_batch_shape = z.shape[:-self.z_ndims]
        x_batch_shape = () if x is None else x.shape[:-self.x_ndims]
        t_batch_shape = () if t is None else jnp.asarray(t).shape
        batch_shape = jnp.broadcast_shapes(z_batch_shape, x_batch_shape, t_batch_shape)

        z = jnp.broadcast_to(z, batch_shape + self.z_shape)
        x = None if x is None else jnp.broadcast_to(x, batch_shape + x.shape[-self.x_ndims:])
        t = None if t is None else jnp.broadcast_to(t, batch_shape)
        return z, x, t

    @property
    def z_shape(self) -> Tuple[int, ...]:
        """Effective z_shape from config."""
        return self.config.main["latent_shape"]
    
    @property
    def z_ndims(self) -> int:
        """Number of dimensions in z_shape."""
        return len(self.z_shape)
    
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

    @property
    def y_ndims(self) -> int:
        """Number of dimensions in y_shape."""
        return len(self.config.main["output_shape"])


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
    
    @nn.compact
    def get_alpha_bar_gamma_prime_t(self, t: jnp.ndarray):
        """Get noise schedule output using @nn.compact method."""
        # Use the noise schedule initialized in setup()
        # The new interface returns (alpha_bar, gamma_prime)
        # Note: stop_gradient is now handled inside the NoiseSchedule class itself
        # based on the learnable field, so we don't need to apply it here anymore
        alpha_bar_t, gamma_prime_t = self.noise_schedule.get_alpha_bar_gamma_prime(t)
        return alpha_bar_t, gamma_prime_t
    
    @nn.compact
    def crn_output(self, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """Raw model output from CRN.  Called u_y in paper"""
        return self.crn_model(z, x, t, training=training)
        
    @nn.compact
    def encoder(self, x: jnp.ndarray, training: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Encoder that maps x to latent space using single factory function.
        
        Returns:
            Tuple of (mu, logvar) where:
            - For normal encoders: encoder returns (mu, logvar) directly
            - For deterministic encoders: encoder returns z, we return (z, -jnp.inf)
        """
        # Use direct factory function: create_encoder(config_dict, input_shape, latent_shape)
        encoder = create_encoder(
            self.config.encoder,
            input_shape=(x.shape[-1],),  # Use flattened input dimension
            latent_shape=self.z_shape  # Use structured latent shape
        )

        # Get encoder output - could be tuple (mu, logvar) or single tensor z
        encoder_output = encoder(x, training)
        
        # Optimized: move assertions outside JIT compilation for better performance
        # (Assertions are kept for debugging but will be compiled out in optimized builds)
        
        # Handle both normal and deterministic cases
        if isinstance(encoder_output, tuple):
            # Normal encoder returns (mu, logvar) tuple
            return encoder_output
        else:
            # Deterministic encoder returns z, we return (z, -jnp.inf) for consistency
            return encoder_output, -jnp.inf

    @nn.compact
    def decoder(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """Decoder that maps latent z to output space using single factory function."""
        # Create decoder network using direct factory
        decoder = create_decoder(
            self.config.decoder,
            latent_shape=self.z_shape,  # Use structured latent shape
            output_shape=self.config.main["output_shape"]  # Use full output_shape
        )
        return decoder(x, training)

    @nn.compact
    def dz_dt(self, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """Flow model that computes dz/dt using CRN with CT-style vector field."""
        # Optimized: minimize broadcasting and avoid redundant operations
        t = jnp.asarray(t)
        z, x, t = self._broadcast_inputs(z, x, t)

        crn_output = self.crn_output(z, x, t, training=training)
        # Get alpha_bar and gamma_prime directly from noise schedule
        t_expanded = jnp.expand_dims(jnp.asarray(t), axis=tuple(range(-self.z_ndims, 0)))
        alpha_t, tau_inverse = self.get_alpha_bar_gamma_prime_t(t_expanded)
        return self.dz_dt_lazy(z, alpha_t, tau_inverse, crn_output)
    
    def dz_dt_lazy(self, z, alpha_t, tau_inverse, crn_output):
        """Lazy version of dz_dt that takes precomputed CRN output and noise schedule parameters."""
        return tau_inverse * (jnp.sqrt(alpha_t) * crn_output - 0.5*(1 + alpha_t) * z)


    def __call__(self, x: jnp.ndarray, y: jnp.ndarray, key: jr.PRNGKey, training: bool = True) -> jnp.ndarray:
        # For initialization, we need to call the nn compact methods to initialize parameters

        batch_shape = y.shape[:-self.y_ndims]
                
        # Call flow_model to initialize its parameters (need dummy z and t)
        dummy_z = jnp.zeros(batch_shape + self.z_shape)
        dummy_t = jnp.zeros(batch_shape)
        
        # Call flow_model to initialize the CRN model parameters and alpha_bar_gamma_prime_t
        dz_dt = self.dz_dt(dummy_z, x, dummy_t, training)
        
        # Call encoder and decoder to initialize their parameters
        # These @nn.compact methods need to be called during initialization to create parameters
        encoder_output = self.encoder(y, training)
        decoder_output = self.decoder(dummy_z, training)
        
        # For initialization, we just return a dummy output
        # The actual forward pass logic is handled by the individual methods
        return jnp.zeros(batch_shape + self.z_shape)

# JIT with training static to avoid traced bool in Dropout
    @partial(jax.jit, static_argnums=(0, 5))
    def loss(self, params: dict, x: jnp.ndarray, y: jnp.ndarray, key: jr.PRNGKey, training: bool = True) -> Tuple[jnp.ndarray, dict]:
        """
        Compute the CT-style SNR-weighted loss.
        """
        # Get target encoding from y encoder with automatic RNG handling
        key, t_key, noise_key, z_target_key = jr.split(key, 4)
        batch_shape = y.shape[:-self.y_ndims]
        # Encode Target (noisy latent)  
        mu_z_target, logvar_z_target = self.apply(params, y, method='encoder', training=training, rngs={'dropout': key})
        z_target = mu_z_target + jnp.exp(0.5 * logvar_z_target) * jr.normal(z_target_key, mu_z_target.shape)
                

        # Sample time and get noise schedule parameters and noise latent at time t
        t = jr.uniform(t_key, batch_shape + self.z_ndims*(1,), minval=0.0, maxval=1.0)
        alpha_bar_t, gamma_prime_t = self.apply(params, t, method='get_alpha_bar_gamma_prime_t')
        z_t = jnp.sqrt(alpha_bar_t) * z_target +  jnp.sqrt(1.0 - alpha_bar_t) * jr.normal(noise_key, z_target.shape)
        
        t = t.squeeze(tuple(range(-self.z_ndims, 0)))
        z_target_est = self.apply(params, z_t, x, t, method='crn_output', training=training, rngs={'dropout': key})
        y_pred = self.apply(params, z_target_est, method='decoder', training=training, rngs={'dropout': key})


        # SNR weight = gamma_prime * exp(gamma) = gamma_prime * (alpha_bar / (1 - alpha_bar))
        # Since alpha_bar = sigmoid(gamma), exp(gamma) = alpha_bar / (1 - alpha_bar)
        use_snr_weight = self.config.main.get("use_snr_weight", True)
        if use_snr_weight:
            snr_weight = gamma_prime_t * alpha_bar_t / (1.0 - alpha_bar_t)
            snr_weight = snr_weight.squeeze(tuple(range(-self.z_ndims, 0)))
        else:
            snr_weight = 1.0
#        snr_weight_mean = jnp.mean(snr_weight)

        squared_error = jnp.mean((z_target_est - z_target) ** 2, axis=tuple(range(-self.z_ndims, 0)))
        snr_loss = jnp.mean(snr_weight * squared_error)

        dz_dt = self.dz_dt_lazy(z_t, alpha_bar_t, gamma_prime_t, z_target_est)
        reg_loss = jnp.mean(dz_dt**2, axis=tuple(range(-self.z_ndims, 0)))  # has batch_shape
        reg_loss = jnp.mean(snr_weight * reg_loss)

        recon_loss_type = self.config.main.get("recon_loss_type", "mse")
        if recon_loss_type == "cross_entropy":
            recon_loss = jnp.mean(-y * jnp.log(y_pred + 1e-8), axis = tuple(range(-self.y_ndims, 0)))
        elif recon_loss_type == "mse":
            recon_loss = jnp.mean((y - y_pred)**2, axis=tuple(range(-self.y_ndims, 0)))
        else:
            recon_loss = 0.0

        recon_loss = jnp.mean(snr_weight * recon_loss)        
        reg_weight = self.config.main.get("reg_weight", 0.0)
        recon_weight = self.config.main.get("recon_weight", 0.0)
        if recon_loss_type == "none":
            recon_weight = 0.0

        total_loss = snr_loss + recon_weight * recon_loss + reg_weight * reg_loss

        return total_loss, {
            'flow_loss': snr_loss,
            'recon_loss': recon_loss, 
            'reg_loss': reg_loss,
            'total_loss': total_loss
        }

    
    @partial(jax.jit, static_argnums=(0, 3, 4, 5))  # self, num_steps, integration_method, output_type, and training are static arguments
    def predict(self, params: dict, x: jnp.ndarray, num_steps: int = 20, integration_method: str = "midpoint", output_type: str = "end_point", prng_key: jr.PRNGKey = None) -> jnp.ndarray:
        """
        Generate predictions using continuous-time flow integration.
        
        For CT flow, we integrate the learned vector field from t=0 to t=1 to generate
        clean samples. The vector field uses sophisticated noise schedules and SNR weighting.
        
        Args:
            params: Model parameters
            x: Input data [batch_shape + x_shape]
            num_steps: Number of integration steps (default: 20)
            integration_method: Integration method ("euler", "heun", "rk4", "adaptive", "midpoint") (default: "midpoint")
            output_type: Type of output ("end_point" for final prediction, "trajectory" for full trajectory)
            prng_key: Optional PRNG key for generative mode. If provided, samples z_0 from unit normal instead of zero.
            
        Returns:
            If output_type="end_point": Final prediction [batch_shape + y_shape]
            If output_type="trajectory": Full trajectory [num_steps, batch_shape + y_shape]
        """
        
        # Disable gradient tracking through parameters for inference
        params_no_grad = jax.lax.stop_gradient(params)
        
        # Generate initial latent state z_0
        batch_shape = x.shape[:-self.x_ndims]
        if prng_key is not None:
            # Generative mode: sample from unit normal distribution
            z_0 = jr.normal(prng_key, batch_shape + (self.z_dim,))
        else:
            # Regression mode: start from zero
            z_0 = jnp.zeros(batch_shape + (self.z_dim,))  # ode expects vectorized z
        
        # Define the CT vector field: dz/dt = tau_inverse(t) * (sqrt(alpha(t))*model_output - (1+alpha(t))/2*z)
        def vector_field(params, z, x, t):
            z = self._unflatten_z(z)
            dz_dt = self.apply(params, z, x, t, method = 'dz_dt', training=False)
            return self._flatten_z(dz_dt)
        
        # Integrate the ODE from t=0 to t=1 (CT flow process)
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
        return self.apply(params, z, method='decoder', training=False)


    @partial(jax.jit, static_argnums=(0, 3, 4, 5, 6))  # self, num_steps, integration_method, output_type are static arguments
    def sample(self, params: dict, prng_key: jr.PRNGKey, batch_shape: Tuple[int, ...], num_steps: int = 20, integration_method: str = "midpoint", output_type: str = "end_point") -> jnp.ndarray:
        """
        Generate samples using continuous-time flow integration without conditional input.
        
        This method is designed for unconditional generation when x=None. It takes an explicit
        batch_shape parameter to determine how many samples to generate.
        
        Args:
            params: Model parameters
            batch_shape: Shape of the batch (e.g., (32,) for 32 samples)
            num_steps: Number of integration steps (default: 20)
            integration_method: Integration method ("euler", "heun", "rk4", "adaptive", "midpoint") (default: "midpoint")
            output_type: Type of output ("end_point" for final prediction, "trajectory" for full trajectory)
            prng_key: Optional PRNG key for generative mode. If provided, samples z_0 from unit normal instead of zero.
            
        Returns:
            If output_type="end_point": Final samples [batch_shape + y_shape]
            If output_type="trajectory": Full trajectory [num_steps, batch_shape + y_shape]
        """
        # batch_shape should already be a tuple of Python integers from the caller
        # Disable gradient tracking through parameters for inference
        params_no_grad = jax.lax.stop_gradient(params)
        
        # Generate initial latent state z_0 with explicit batch_shape
        z_0 = jr.normal(prng_key, batch_shape + (self.z_dim,))
        
        # Define the CT vector field: dz/dt = tau_inverse(t) * (sqrt(alpha(t))*model_output - (1+alpha(t))/2*z)
        def vector_field(params, z, x, t):
            z = self._unflatten_z(z)
            dz_dt = self.apply(params, z, None, t, method='dz_dt', training=False)  # Use x=None
            return self._flatten_z(dz_dt)
        
        # Integrate the ODE from t=0 to t=1 (CT flow process)
        z = integrate_ode(
            vector_field=vector_field,
            params=params_no_grad,
            z0=z_0,
            x=None,  # No conditional input
            time_span=(0.0, 1.0),
            num_steps=num_steps,
            method=integration_method,
            output_type=output_type
        )
        
        z = self._unflatten_z(z)
        return self.apply(params, z, method='decoder', training=False)


    @partial(jax.jit, static_argnums=(0, 5, 7))  # self, optimizer, training are static
    def train_step(self, params: dict, x: jnp.ndarray, y: jnp.ndarray, opt_state: dict, optimizer, key: jr.PRNGKey, training: bool = True) -> Tuple[dict, dict, jnp.ndarray, dict]:
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
    
    
    @partial(jax.jit, static_argnums=(0, 5))  # self and optimizer are static arguments
    def train_step_without_dropout(self, params: dict, x: jnp.ndarray, y: jnp.ndarray, opt_state: dict, optimizer, key: jr.PRNGKey) -> Tuple[dict, dict, jnp.ndarray, dict]:
        """JIT-compiled training step with dropout disabled."""
        def loss_fn(params):
            return self.loss(params, x, y, key, training=False)
        
        (loss, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss, metrics


