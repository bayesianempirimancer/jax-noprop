import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
from flax.core import FrozenDict
import optax
from typing import Tuple, Dict, Optional
import inspect

from functools import partial, cached_property
import math

# Import directly without going through src package to avoid einops dependency
from src.flow_models.config import Config
from src.vae.encoders import create_encoder
from src.vae.decoders import create_decoder
from src.flow_models.crn import create_conditional_resnet
from src.embeddings.noise_schedules import create_noise_schedule
from src.utils.ode_integration import integrate_ode
from src.layers.settrans import SAB, ISAB, PMA

from jax.scipy.special import logsumexp


class SetTransfromer(nn.Module):
    output_dim: int
    num_heads: int
    embed_dim: int
    induced_dim: int
    seed_dim: int
    
    @nn.compact
    def __call__(self,x):
        x = ISAB(N_head=self.num_heads, N_dim=self.embed_dim, N_induced=self.induced_dim, ln=True)(x)
        x = ISAB(N_head=self.num_heads, N_dim=self.embed_dim, N_induced=self.induced_dim, ln=True)(x)
        x = PMA(N_head=self.num_heads, N_dim=self.embed_dim, N_seed=self.seed_dim, ln=True)(x)
        x = SAB(N_head=self.num_heads, N_dim=self.embed_dim, ln=True)(x)
        return nn.DenseGeneral(self.output_dim, axis=(-2,-1),
                            kernel_init =  nn.initializers.variance_scaling(scale = 1/3,
                                                                            mode = "fan_in",
                                                                            distribution = "uniform"))(x)

class MixtureComponents(nn.Module):
    """This class implements sampling from ammortized posterior over components q(c|x),
    and computes the KL divergence between q(c|x) and a prior p(c|y)
    """
    num_components: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, key: jr.PRNGKey, num_samples: int, y: Optional[jnp.ndarray] = None) -> jnp.ndarray:

        logits_prior = nn.Dense(self.num_components, kernel_init=jax.nn.initializers.xavier_normal())(x)

        if y is not None:
            logits_posterior = SetTransfromer(
                output_dim=self.num_components,
                num_heads=4,
                embed_dim=64,
                induced_dim=16,
                seed_dim=4
            )(y)
            log_prob_post = logits_posterior - logsumexp(logits_posterior, axis=-1, keepdims=True)
            log_prob_prior = logits_prior - logsumexp(logits_prior, axis=-1, keepdims=True)
            kl_div = jnp.sum( jnp.exp(log_prob_post) * (log_prob_post - log_prob_prior), -1).mean()
        else:
            logits_posterior = logits_prior
            kl_div = 0.0

        c = jnp.argmax(logits_posterior + jr.gumbel(key, shape=logits_posterior.shape), axis=-1, keepdims=True)
        return jax.nn.one_hot(c, self.num_components).repeat(num_samples, axis=1), kl_div

class VAE_flow_mix(nn.Module):
    """Variational Autoencoder with flow model using @nn.compact methods."""
    config: Config
    
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
        
        # Store config values as instance variables for use in JIT-compiled functions
        self.normalize_snr_weight = bool(self.config.main.get("normalize_snr_weight", False))
        self.recon_loss_type = self.config.main.get("recon_loss_type", "mse")
        self.recon_weight = float(self.config.main.get("recon_weight", 0.0))
        self.reg_weight = float(self.config.main.get("reg_weight", 0.0))
        self.vae_weight = float(self.config.main.get("vae_weight", 0.0))
        
        # Create schedule using factory - pass learnable flag to schedule
        # The schedule will handle stop_gradient internally if learnable=False
        # Schedule classes use their own defaults for parameters
        self.no_noise_shedule = bool(self.config.main.get('no_noise_schedule', False))
        if not self.no_noise_shedule:
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

        num_components = int(self.config.main.get('num_components', 10))
        self.mixture = MixtureComponents(num_components)
    

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
    def dz_dt(self, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray, x_mask: Optional[jnp.ndarray] = None, training: bool = True) -> jnp.ndarray:
        """Flow model that computes dz/dt using CRN."""
        # Optimized: minimize broadcasting and avoid redundant operations
        t = jnp.asarray(t)
        z, x, t = self._broadcast_inputs(z, x, t)
        
        # For seq to seq models we must encode x in the same manner as y before passing to seq_crn
        if self.config.main.get("encode_x", False) and x is not None:
            x = self.encode(x, training)[0]
        
        # Pass x_mask to CRN only if the model supports it (sequence/point cloud models)
        # Check if the __call__ method accepts x_mask parameter
        call_sig = inspect.signature(self.crn_model.__call__)
        if 'x_mask' in call_sig.parameters:
            dz_dt = self.crn_model(z, x, t, x_mask=x_mask, training=training)            
        else:
            dz_dt = self.crn_model(z, x, t, training=training)            
        return dz_dt

    def lazy_flow(self, z_t, z_target, alpha_t, gamma_prime_t):
        return gamma_prime_t * (jnp.sqrt(alpha_t)* z_target - (0.5*(1.0 + alpha_t)) * z_t)

    # KL divergence with SNR weighting: SNR_noise_weight = gamma_prime
    # Note:  0.5 is dropped and the proof is here: https://arxiv.org/html/2312.10393v1
    def lazy_snr_weight(self, alpha_t, gamma_prime_t):
        # SNR weight: SNR'(t) = gamma_prime * alpha / (1-alpha)
        return self.lazy_flow_snr(alpha_t, gamma_prime_t)
    
    def lazy_noise_snr(self, alpha_t, gamma_prime_t):  return gamma_prime_t
    def lazy_target_snr(self, alpha_t, gamma_prime_t): return gamma_prime_t * alpha_t / (1.0 - alpha_t)
    def lazy_error_snr(self, alpha_t, gamma_prime_t):  return gamma_prime_t / (1.0 - alpha_t)
    def lazy_score_snr(self, alpha_t, gamma_prime_t):  return gamma_prime_t * (1.0 - alpha_t)
    def lazy_flow_snr(self, alpha_t, gamma_prime_t):   return  1.0 / ((1.0 - alpha_t) * alpha_t * gamma_prime_t)

    def lazy_score(self, z_t, z_target, alpha_t):     return (jnp.sqrt(alpha_t)*z_target - z_t) / (1.0 - alpha_t)
    def lazy_error(self, z_t, z_target, alpha_t):     return z_t - jnp.sqrt(alpha_t)* z_target
    def lazy_noise(self, z_t, z_target, alpha_t):     return self.lazy_error(z_t, z_target, alpha_t) / jnp.sqrt(1.0 - alpha_t)
    
    @nn.compact
    def get_component(self, x: jnp.ndarray, key: jr.PRNGKey, num_samples: int, y: jnp.ndarray = None):
        return self.mixture(x, key, num_samples, y=y)

    @nn.compact
    def get_noise_params(self, t: jnp.ndarray):
        """Get noise schedule output using @nn.compact method."""
        alpha_bar_t, gamma_prime_t = self.noise_schedule.get_alpha_bar_gamma_prime(t)
        return alpha_bar_t, gamma_prime_t
    
    @nn.compact
    def get_alpha_bar(self, t: jnp.ndarray):
        """Get alpha_bar(t) from noise schedule using @nn.compact method."""
        return self.noise_schedule.get_alpha_bar(t)
    
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
        batch_shape = (math.prod(y.shape[:-self.y_ndims]),)
        # Call flow_model to initialize its parameters (need dummy z and t)
        dummy_z = jnp.zeros(batch_shape + self.z_shape)
        dummy_t = jnp.zeros(batch_shape)
        
        # Call get_noise_params to initialize noise schedule parameters
        if not self.no_noise_shedule:
            self.get_noise_params(dummy_t)
            self.get_alpha_bar(dummy_t)

        dummy_c, _ = self.mixture(x, key, y.shape[1], y=y)
        
        # initialize model components
        flow_output = self.dz_dt(dummy_z, dummy_c.reshape(*batch_shape, -1), dummy_t, training)
        encoder_output = self.encode(y.reshape(*batch_shape, -1), training)
        decoder_output = self.decode(dummy_z, training)
        
        return jnp.zeros(batch_shape + self.z_shape) # batch consistent with expectations

    @partial(jax.jit, static_argnums=(0, 5))    
    def loss(self, params: dict, x: Optional[jnp.ndarray], y: jnp.ndarray, key: jr.PRNGKey, training: bool = True, x_mask: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, dict]:
        """
        Compute the loss by calling individual @nn.compact methods with proper rngs.
        """
        # Extract config values at the start (self is static, so this is safe)
        # Convert to concrete Python types to avoid tracing issues
        normalize_snr_weight = bool(self.config.main.get("normalize_snr_weight", False))
        recon_loss_type = str(self.config.main.get("recon_loss_type", "mse"))
        recon_weight = float(self.config.main.get("recon_weight", 0.0))
        reg_weight = float(self.config.main.get("reg_weight", 0.0))
        vae_weight = float(self.config.main.get("vae_weight", 1.0))
        use_noise_shedule = not bool(self.config.main.get('no_noise_schedule', False))
        
        # Split keys for random sampling operations (not dropout)
        key, t_key, z_0_key, z_t_noise_key, z_target_key, vae_noise_key, mixture_key = jr.split(key, 7)
        batch_shape = (math.prod(y.shape[:-self.y_ndims]),)        

        # Encode Target (noisy latent)
        mu_z_target, logvar_z_target = self.apply(params, y.reshape(*batch_shape, -1), method='encode', training=training, rngs={'dropout': key})
        z_target = mu_z_target + jr.normal(z_target_key, mu_z_target.shape) * jnp.exp(0.5 * logvar_z_target)

        # Sample initial latent state and time
        t = jr.uniform(t_key, batch_shape, minval=0.0, maxval=1.0)
        t_expanded = jnp.expand_dims(t, axis=tuple(range(-self.z_ndims, 0)))
        z_0 = jr.normal(z_0_key, batch_shape + self.z_shape)
        
        # Sample latent state at time t
        diff_z = z_target - z_0
        z_t = z_0 + t_expanded * diff_z

        if use_noise_shedule:
            # Get noise schedule parameters (use squeezed t)
            alpha_t, gamma_prime_t = self.apply(params, t, method='get_noise_params')
            # Squeeze alpha_t and gamma_prime_t for use in SNR weight computation
            alpha_t_squeezed = jnp.squeeze(alpha_t, axis=tuple(range(-self.z_ndims, 0))) if alpha_t.ndim > len(batch_shape) else alpha_t
            gamma_prime_t_squeezed = jnp.squeeze(gamma_prime_t, axis=tuple(range(-self.z_ndims, 0))) if gamma_prime_t.ndim > len(batch_shape) else gamma_prime_t
        
        c, kl_div_comp = self.apply(params, x, mixture_key, num_samples=y.shape[1], y=y, method='get_component')

        # Compute Flow Field and Target Estimate
        dz_dt = self.apply(params, z_t, c.reshape(*batch_shape, -1), t, x_mask, method='dz_dt', training=training, rngs={'dropout': key})    

        # Compute Predictions
        if recon_weight > 0.0:
            z_target_est = dz_dt * (1.0-t_expanded) + z_t
            y_pred = self.apply(params, z_target_est, method='decode', training=training, rngs={'dropout': key}).reshape(y.shape)
        if vae_weight > 0.0:
            # get \bar{\alpha} at t=1.0
            alpha_1 = self.apply(params, jnp.array(1.0), method='get_alpha_bar')
            z_target_vae = z_target*jnp.sqrt(alpha_1) + jr.normal(vae_noise_key, z_target.shape) * jnp.sqrt(1.0 - alpha_1)
            y_vae = self.apply(params, z_target_vae, method='decode', training=training, rngs={'dropout': key}).reshape(y.shape)

        # Compute Losses
        if use_noise_shedule:
            snr_weight = self.lazy_flow_snr(alpha_t_squeezed, gamma_prime_t_squeezed)
            # Normalize SNR weights by their mean if normalize_snr_weight is True
            if normalize_snr_weight:
                snr_weight_mean = jnp.mean(snr_weight)
                snr_weight = snr_weight / (snr_weight_mean + 1e-8)
        else:
            snr_weight = 1.0

        squared_error = jnp.mean((dz_dt - diff_z)**2, axis=tuple(range(-self.z_ndims, 0)))
        flow_loss = jnp.mean(snr_weight * squared_error)

        if reg_weight > 0.0:
            reg_loss = jnp.mean(dz_dt**2)
        else:
            reg_loss = 0.0

        vae_loss = 0.0
        recon_loss = 0.0
        if recon_loss_type == "cross_entropy":
            if recon_weight > 0.0:
                recon_loss = optax.losses.safe_softmax_cross_entropy(y_pred, y)
            if vae_weight > 0.0:
                vae_loss   = optax.losses.safe_softmax_cross_entropy(y_vae, y)
        elif recon_loss_type == "mse":
            if recon_weight > 0.0:
                recon_loss = jnp.sum((y - y_pred)**2, axis=tuple(range(-self.y_ndims, 0))).reshape(-1)
            if vae_weight > 0.0:
                vae_loss   = jnp.sum((y - y_vae)**2, axis=tuple(range(-self.y_ndims, 0))).reshape(-1)

        if use_noise_shedule and recon_weight > 0.0:
            recon_snr = self.lazy_target_snr(alpha_t_squeezed, gamma_prime_t_squeezed)
            # Normalize recon SNR weights by their mean if normalize_snr_weight is True
            if normalize_snr_weight:
                recon_snr_mean = jnp.mean(recon_snr)
                recon_snr = recon_snr / (recon_snr_mean + 1e-8)
        else:
            recon_snr = 1.0
        
        recon_loss = jnp.mean(recon_snr * recon_loss)  # Average over batch dimension if needed      
        vae_loss = jnp.mean(vae_loss)

        # KL regularization term: KL(q(z_0|z_target), p(z_0))
        # alpha_0 is guaranteed to be in (0, 1) by noise schedule clipping
        # alpha_0 = self.apply(params, jnp.array(0.0), method='get_alpha_bar')
        # q_sigma_sq = 1.0 - alpha_0
        # mean_q_mu_sq_over_sigma_sq = alpha_0/q_sigma_sq*jnp.mean(jnp.sum(z_target**2, axis=tuple(range(-self.z_ndims, 0))))
        # kl_z0_loss = 0.5 * (mean_q_mu_sq_over_sigma_sq + self.z_dim*(jnp.log(q_sigma_sq) - 1.0))


        total_loss = flow_loss + recon_weight * recon_loss + reg_weight * reg_loss + vae_weight * vae_loss + kl_div_comp # + kl_z0_loss  
        
        return total_loss, {
            'flow_loss': flow_loss,
            'recon_loss': recon_loss, 
            'reg_loss': reg_loss,
            'vae_loss': vae_loss,
            'kl_z0_loss': 0.0, # kl_z0_loss,
            'total_loss': total_loss
        }


    @partial(jax.jit, static_argnums=(0, 3, 4, 5))  # self, num_steps, integration_method, output_type are static arguments
    def predict(self, params: dict, x: jnp.ndarray, num_steps: int = 20, integration_method: str = "euler", output_type: str = "end_point", prng_key: jr.PRNGKey = None) -> jnp.ndarray:
        """
        Make predictions using ODE solver integration.        
        Requires x is not None... use sample method for unconditional generation.
        """
        num_samples = 64
        params_no_grad = jax.lax.stop_gradient(params)
        x_batch_shape = x.shape[:-self.x_ndims]
        batch_shape = (math.prod(x_batch_shape + (num_samples,)), )
        
        if prng_key is not None:
            # Generative mode: sample from unit normal distribution
            key1, key2 = jr.split(prng_key)
            z_0 = jr.normal(key1, batch_shape + (self.z_dim,))
            c, _ = self.apply(params, x, key2, num_samples, method='get_component')
        else:
            # Regression mode: start from zero
            z_0 = jnp.zeros(batch_shape + (self.z_dim,))  # ode expects vectorized z
            c, _ = self.apply(params, x, jr.PRNGKey(0), num_samples, method='get_component')
        
        def vector_field(params, z, x, t):
            z = self._unflatten_z(z)  # crn expects unflattened z
            dz_dt = self.apply(params, z, x, t, method='dz_dt', training=False)
            return self._flatten_z(dz_dt)  # ODE solver expects flattened z
        
        # Integrate the ODE from t=0 to t=1
        z = integrate_ode(
            vector_field=vector_field,
            params=params_no_grad,
            z0=z_0,
            x=c.reshape(*batch_shape, -1),
            time_span=(0.0, 1.0),
            num_steps=num_steps,
            method=integration_method,
            output_type=output_type
        )
        z = self._unflatten_z(z)
        return self.apply(params, z, method='decode', training=False).reshape(x_batch_shape + (num_samples, -1))
    
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
        key1, key2 = jr.split(prng_key)
        z_0 = jr.normal(key1, batch_shape + (self.z_dim,))
        c = self.apply(params, None, key2, method='get_component', training=False)

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
            x=c,  # No conditional input
            time_span=(0.0, 1.0),
            num_steps=num_steps,
            method=integration_method,
            output_type=output_type
        )

        return self.apply(params, z_trajectory, method='decode', training=False)
    

    @partial(jax.jit, static_argnums=(0, 5, 7))  # self, optimizer, and training are static arguments
    def train_step(self, params: dict, x: Optional[jnp.ndarray], y: jnp.ndarray, opt_state: dict, optimizer, key: jr.PRNGKey, training: bool = True, x_mask: Optional[jnp.ndarray] = None) -> Tuple[dict, dict, jnp.ndarray, dict]:
        """
        JIT-compiled training step for VAE with flow model.
        
        Args:
            params: Current model parameters
            x: Conditional input [batch_size, input_dim] or [batch_size, seq_len, embed_dim] for sequences
            y: Target output [batch_size, output_dim] or [batch_size, seq_len, embed_dim] for sequences
            opt_state: Optimizer state
            optimizer: Optax optimizer
            key: Random key
            training: Whether in training mode
            x_mask: Boolean mask [batch_size, x_seq_len] for sequence models (True=valid, False=masked)
            
        Returns:
            params: Updated model parameters
            opt_state: Updated optimizer state
            loss: Training loss
            metrics: Training metrics
        """
        # Compute loss and gradients
        def loss_fn(params):
            return self.loss(params, x, y, key, training=training, x_mask=x_mask)  
        
        (loss, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        
        # Update parameters using optimizer
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss, metrics
    
    @partial(jax.jit, static_argnums=(0, 5, 7))  # self and optimizer are static arguments
    def train_step_without_dropout(self, params: dict, x: jnp.ndarray, y: jnp.ndarray, opt_state: dict, optimizer, key: jr.PRNGKey, training: bool = False, x_mask: Optional[jnp.ndarray] = None) -> Tuple[dict, dict, jnp.ndarray, dict]:
        # Compute loss and gradients
        def loss_fn(params):
            return self.loss(params, x, y, key, training=training, x_mask=x_mask)  
        
        (loss, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        
        # Update parameters using optimizer
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss, metrics