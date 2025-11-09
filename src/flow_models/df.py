import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
from flax.core import FrozenDict
import optax
from typing import Tuple, Dict, Optional

from functools import partial, cached_property

# Import directly without going through src package to avoid einops dependency
from src.flow_models.config import Config
from src.vae.encoders import create_encoder
from src.vae.decoders import create_decoder
from src.flow_models.crn import create_conditional_resnet
from src.embeddings.noise_schedules import create_noise_schedule
from src.utils.ode_integration import integrate_ode


class VAE_flow(nn.Module):
    """Variational Autoencoder with flow model using @nn.compact methods."""
    config: Config
    
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
        
        # Store config values as instance variables for use in JIT-compiled functions
        self.normalize_snr_weight = bool(self.config.main.get("normalize_snr_weight", False))
        self.recon_loss_type = self.config.main.get("recon_loss_type", "mse")
        self.recon_weight = float(self.config.main.get("recon_weight", 0.0))
        self.reg_weight = float(self.config.main.get("reg_weight", 0.0))
        
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
    def get_noise_params(self, t: jnp.ndarray):
        """Get noise schedule output using @nn.compact method."""
        # Use the noise schedule initialized in setup()
        # The new interface returns (alpha_bar, gamma_prime)
        # Note: stop_gradient is now handled inside the NoiseSchedule class itself
        # based on the learnable field, so we don't need to apply it here anymore
        alpha_bar_t, gamma_prime_t = self.noise_schedule.get_alpha_bar_gamma_prime(t)
        return alpha_bar_t, gamma_prime_t
    
    @nn.compact
    def pred_noise(self, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """Flow model that computes dz/dt using CRN."""
        # Optimized: minimize broadcasting and avoid redundant operations
        t = jnp.asarray(t)
        z, x, t = self._broadcast_inputs(z, x, t)
        
        # Encode x to latent space before passing to CRN (if encode_x is True)
        # encode() always returns a tuple (mu, logvar) or (z, -jnp.inf)
        if self.config.main.get("encode_x", False) and x is not None:
            x = self.encode(x, training)[0]
                
        # Use the CRN model created in setup()
        pred_noise = self.crn_model(z, x, t, training=training)        
        return pred_noise
    
    @nn.compact
    def encode(self, x: jnp.ndarray, training: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Encoder that maps x to latent space."""
        encoder_output = self.encoder(x, training)
        if isinstance(encoder_output, tuple):
            return encoder_output              # Normal encoder returns (mu, logvar) tuple
        else:
            return encoder_output, -jnp.inf    # Deterministic encoder returns z, we return (z, -jnp.inf) for consistency

    @nn.compact
    def decode(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """Decoder that maps latent z to output space."""
        return self.decoder(x, training)

    @nn.compact
    def dz_dt(self, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Compute dz/dt (diffusion vector field) for given z, x, t.
        For diffusion: dz/dt = 0.5*z - predicted_noise
        
        Uses noise schedule: alpha = sigmoid(gamma(t)), alpha_prime = gamma_prime(t)*alpha*(1-alpha)
        Usual Equation: dz_dt = 0.5*alpha_prime/alpha * z - predicted_noise*alpha_prime/alpha/sqrt(1-alpha)
        """
        predicted_noise = self.pred_noise(z, x, t, training=training)
        t = jnp.expand_dims(jnp.asarray(t), axis=tuple(range(-self.z_ndims, 0)))
        alpha, gamma_prime = self.get_noise_params(t)
        return self.lazy_flow(z, predicted_noise, alpha, gamma_prime)

    def lazy_flow(self, z_t, predicted_noise, alpha_t, gamma_prime_t):
        return gamma_prime_t * (0.5*(1-alpha_t)*z_t - jnp.sqrt(1-alpha_t)*predicted_noise)

    # KL divergence with SNR weighting: SNR_noise_weight = gamma_prime
    # Note:  0.5 is dropped and the proof is here: https://arxiv.org/html/2312.10393v1
    def lazy_snr_weight(self, alpha_t, gamma_prime_t):
        # Diffusion ModelSNR weight
        return self.lazy_noise_snr(alpha_t, gamma_prime_t)
    
    def lazy_noise_snr(self, alpha_t, gamma_prime_t):  return gamma_prime_t
    def lazy_target_snr(self, alpha_t, gamma_prime_t): return gamma_prime_t * alpha_t / (1.0 - alpha_t)
    def lazy_error_snr(self, alpha_t, gamma_prime_t):  return gamma_prime_t / (1.0 - alpha_t)
    def lazy_score_snr(self, alpha_t, gamma_prime_t):  return gamma_prime_t * (1.0 - alpha_t)
    def lazy_flow_snr(self, alpha_t, gamma_prime_t):   return  1.0 / ((1.0 - alpha_t) * alpha_t * gamma_prime_t)

    def lazy_score(self, z_t, z_target, alpha_t):     return (jnp.sqrt(alpha_t)*z_target - z_t) / (1.0 - alpha_t)
    def lazy_error(self, z_t, z_target, alpha_t):     return z_t - jnp.sqrt(alpha_t)* z_target
    def lazy_noise(self, z_t, z_target, alpha_t):     return self.lazy_error(z_t, z_target, alpha_t) / jnp.sqrt(1.0 - alpha_t)

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray, key: jr.PRNGKey, training: bool = True) -> jnp.ndarray:
        # For initialization, we need to call the nn compact methods to initialize parameters

        batch_shape = y.shape[:-self.y_ndims]
                
        # Call pred_noise to initialize its parameters (need dummy z and t)
        dummy_z = jnp.zeros(batch_shape + self.z_shape)
        dummy_t = jnp.zeros(batch_shape)
        
        # Call get_noise_params to initialize noise schedule parameters
        self.get_noise_params(dummy_t)
        
        # Call pred_noise to initialize the CRN model parameters
        flow_output = self.pred_noise(dummy_z, x, dummy_t, training)
        
        # Call encoder and decoder to initialize their parameters
        # These @nn.compact methods need to be called during initialization to create parameters
        encoder_output = self.encode(y, training)
        decoder_output = self.decode(dummy_z, training)
        
        # For initialization, we just return a dummy output
        # The actual forward pass logic is handled by the individual methods
        return jnp.zeros(batch_shape + self.z_shape)

    @partial(jax.jit, static_argnums=(0, 5))  # self, num_steps, integration_method, output_type, and training are static arguments
    def loss(self, params: dict, x: Optional[jnp.ndarray], y: jnp.ndarray, key: jr.PRNGKey, training: bool = True) -> Tuple[jnp.ndarray, dict]:
        """
        Compute the diffusion loss.
        
        For diffusion, the loss is MSE between predicted noise and actual noise:
        L_diff = E[||model_output - noise||²/beta], where beta is 1/t for the linear noise schedule.
        """
        # Extract config values at the start (self is static, so this is safe)
        # Convert to concrete Python types to avoid tracing issues
        normalize_snr_weight = bool(self.config.main.get("normalize_snr_weight", False))
        recon_loss_type = str(self.config.main.get("recon_loss_type", "mse"))
        recon_weight = float(self.config.main.get("recon_weight", 0.0))
        reg_weight = float(self.config.main.get("reg_weight", 0.0))
        vae_weight = float(self.config.main.get("vae_weight", 1.0))
        
        # Split keys for random sampling operations (not dropout)
        key, t_key, noise_key, z_target_key, vae_noise_key = jr.split(key, 5)
        batch_shape = y.shape[:-self.y_ndims]

        alpha_0 = self.apply(params, jnp.asarray(1e-6), method='get_noise_params')[0]
        alpha_1 = self.apply(params, jnp.asarray(1-1e-6), method='get_noise_params')[0]

        # Encode Target (noisy latent)
        mu_z_target, logvar_z_target = self.apply(params, y, method='encode', training=training, rngs={'dropout': key})
        z_target = mu_z_target + jr.normal(z_target_key, mu_z_target.shape) * jnp.exp(0.5 * logvar_z_target)
                        
        # Sample noise and time
        t = jr.uniform(t_key, batch_shape, minval=0.0, maxval=1.0)
        noise = jr.normal(noise_key, z_target.shape)

        # Get noise schedule parameters or z manipulation
        t_expanded = jnp.expand_dims(t, axis=tuple(range(-self.z_ndims, 0)))
        alpha_t, gamma_prime_t = self.apply(params, t_expanded, method='get_noise_params')
        
        sqrt_alpha_t = jnp.sqrt(alpha_t)
        sqrt_1_minus_alpha_t = jnp.sqrt(1.0 - alpha_t)

        # Sample Latent state 
        z_t =  sqrt_alpha_t* z_target +  sqrt_1_minus_alpha_t * noise # noisy latent

        # Compute Predicted Noise and Target Estimate and FLow Estimate
        predicted_noise = self.apply(params, z_t, x, t, method='pred_noise', training=training, rngs={'dropout': key})
        z_target_est = (z_t - predicted_noise * sqrt_1_minus_alpha_t)/(sqrt_alpha_t)
        dz_dt = self.lazy_flow(z_t, predicted_noise, alpha_t, gamma_prime_t)

        # Compute Predictions
        y_pred = self.apply(params, z_target_est, method='decode', training=training, rngs={'dropout': key})
        z_target_vae = z_target*jnp.sqrt(alpha_1) + jr.normal(vae_noise_key, z_target.shape) * jnp.sqrt(1.0 - alpha_1)
        y_vae = self.apply(params, z_target_vae, method='decode', training=training, rngs={'dropout': key})

        # Squeeze alpha_t and gamma_prime_t for use in SNR weight computation
        alpha_t = jnp.squeeze(alpha_t, axis=tuple(range(-self.z_ndims, 0)))
        gamma_prime_t = jnp.squeeze(gamma_prime_t, axis=tuple(range(-self.z_ndims, 0)))
        snr_weight = self.lazy_noise_snr(alpha_t, gamma_prime_t)
        # Normalize SNR weights by their mean if normalize_snr_weight is True
        if normalize_snr_weight:
            snr_weight_mean = jnp.mean(snr_weight)
            snr_weight = snr_weight / (snr_weight_mean + 1e-8)

        # Compute Losses
        squared_error = jnp.mean((noise - predicted_noise) ** 2, axis=tuple(range(-self.z_ndims, 0)))
        flow_loss = jnp.mean(snr_weight * squared_error)
        reg_loss = jnp.mean(dz_dt**2) 

        if recon_loss_type == "cross_entropy":
            recon_loss = jnp.sum(-y * jnp.log(y_pred + 1e-8), axis = tuple(range(-self.y_ndims, 0)))
            vae_loss   = jnp.sum(-y * jnp.log(y_vae + 1e-8), axis = tuple(range(-self.y_ndims, 0)))
        elif recon_loss_type == "mse":
            recon_loss = jnp.sum((y - y_pred)**2, axis=tuple(range(-self.y_ndims, 0)))
            vae_loss   = jnp.sum((y - y_vae)**2, axis=tuple(range(-self.y_ndims, 0)))
        else:
            recon_loss = 0.0
        # recon_loss = jnp.mean(self.lazy_target_snr(alpha_t, gamma_prime_t)*recon_loss)  # Average over batch dimension if needed        
        recon_snr = self.lazy_target_snr(alpha_t, gamma_prime_t)
        # Normalize recon SNR weights by their mean if normalize_snr_weight is True
        if normalize_snr_weight:
            recon_snr_mean = jnp.mean(recon_snr)
            recon_snr = recon_snr / (recon_snr_mean + 1e-8)
        recon_loss = jnp.mean(recon_snr * recon_loss)  # Average over batch dimension if needed      

        vae_loss = jnp.mean(vae_loss)

        # KL regularization term: KL(q(z_0|z_target), p(z_0))
        # alpha_0 is guaranteed to be in (0, 1) and alpha_0 should be close to 0 and q_sigma_sq should be close to 1.0
        q_sigma_sq = 1.0 - alpha_0
        mean_q_mu_sq_over_sigma_sq = alpha_0/q_sigma_sq*jnp.mean(jnp.sum(z_target**2, axis=tuple(range(-self.z_ndims, 0))))
        kl_z0_loss = 0.5 * (mean_q_mu_sq_over_sigma_sq + self.z_dim*(jnp.log(q_sigma_sq) - 1.0))

        # y_mu = self.apply(params, mu_z_target, method='decode', training=training, rngs={'dropout': key})
        # direct_recon_Loss = jnp.mean((y - y_mu)**2)

        total_loss = flow_loss + recon_weight * recon_loss + reg_weight * reg_loss + vae_weight * vae_loss + kl_z0_loss
        # total_loss = total_loss/snr_weight_mean
        
        return total_loss, {
            'flow_loss': flow_loss,  # Add separate diffusion_loss metric
            'recon_loss': recon_loss, 
            'reg_loss': reg_loss,
            'vae_loss': vae_loss,
            'kl_z0_loss': kl_z0_loss,
            'total_loss': total_loss
        }

    
    @partial(jax.jit, static_argnums=(0, 3, 4, 5))  # self, num_steps, integration_method, output_type, and training are static arguments
    def predict(self, params: dict, x: jnp.ndarray, num_steps: int = 20, integration_method: str = "midpoint", output_type: str = "end_point", prng_key: jr.PRNGKey = None) -> jnp.ndarray:
        """
        Generate predictions using ODE solver integration
        """
        # Disable gradient tracking through parameters for inference
        params_no_grad = jax.lax.stop_gradient(params)
        batch_shape = x.shape[:-self.x_ndims]

        # Generate flattened initial latent state z_0
        if prng_key is not None:
            z_0 = jr.normal(prng_key, batch_shape + (self.z_dim,))
        else:
            z_0 = jnp.zeros(batch_shape + (self.z_dim,))  # ode expects vectorized z
        
        # Define the diffusion vector field: dz/dt = 0.5*z - predicted_noise
        def vector_field(params, z, x, t):
            z = self._unflatten_z(z)
            dz_dt = self.apply(params, z, x, t, method='dz_dt', training=False)
            return self._flatten_z(dz_dt)
        
        # Integrate the ODE from t=0 to t=1 (reverse diffusion process)
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
    def sample(self, params: dict, prng_key: jr.PRNGKey, batch_shape: Tuple[int, ...], num_steps: int = 20, integration_method: str = "midpoint", output_type: str = "end_point") -> jnp.ndarray:
        """
        Generate samples using diffusion sampling without conditional input.
        Returns:
            If output_type="end_point": Final samples [batch_shape + y_shape]
            If output_type="trajectory": Full trajectory [num_steps, batch_shape + y_shape]
        """        
        params_no_grad = jax.lax.stop_gradient(params)
        z_0 = jr.normal(prng_key, batch_shape + (self.z_dim,))
        # Define the vector field for ODE integration using flow_model method with x=None
        def vector_field(params, z, x, t):
            z = self._unflatten_z(z)
            dz_dt = self.apply(params, z, None, t, method='dz_dt', training=False)  # Use x=None
            return self._flatten_z(dz_dt)
        
        # Integrate the ODE from t=0 to t=1 (reverse diffusion process)
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
        return self.apply(params, z, method='decode', training=False)


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