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
from src.flow_models.crns.crn import create_conditional_resnet
from src.embeddings.noise_schedules import create_noise_schedule
from src.utils.ode_integration import integrate_ode


class VAE_flow(nn.Module):
    """Variational Autoencoder with continuous-time flow model using @nn.compact methods.
    
    This class implements a VAE with a continuous-time flow model that uses sophisticated
    noise schedules and SNR-weighted loss functions. It combines the stability features
    from df.py with the advanced CT features from ct_orig.py.
    """
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

        alpha_t, gamma_prime_t = self.noise_schedule.get_alpha_bar_gamma_prime(t)
        return alpha_t, gamma_prime_t
    
    @nn.compact
    def get_alpha_bar(self, t: jnp.ndarray):
        """Get alpha_bar(t) from noise schedule using @nn.compact method."""
        return self.noise_schedule.get_alpha_bar(t)
    
    @nn.compact
    def lazy_target_snr(self, alpha_t: jnp.ndarray, gamma_prime_t: jnp.ndarray) -> jnp.ndarray:
        """Lazy routine for target SNR - no params needed."""
        return self.noise_schedule.lazy_target_snr(alpha_t, gamma_prime_t)

    @nn.compact
    def lazy_flow_from_target(self, z: jnp.ndarray, target_estimate: jnp.ndarray, alpha_t: jnp.ndarray, gamma_prime_t: jnp.ndarray) -> jnp.ndarray:
        """Lazy routine for flow from target - no params needed."""
        return self.noise_schedule.lazy_flow_from_target(z, target_estimate, alpha_t, gamma_prime_t)

    
    @nn.compact
    def crn_output(self, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """Raw model output from CRN.  Called u_y in paper"""
        # Encode x to latent space before passing to CRN (if encode_x is True)
        # encode() always returns a tuple (mu, logvar) or (z, -jnp.inf)
        if self.config.main.get("encode_x", False) and x is not None:
            x = self.encode(x, training)[0]
        return self.crn_model(z, x, t, training=training)
        
    @nn.compact
    def encode(self, x: jnp.ndarray, training: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Encoder that maps x to latent space."""
        encoder_output = self.encoder(x, training)
        if isinstance(encoder_output, tuple):
            return encoder_output   # Normal encoder returns (mu, logvar) tuple
        else:
            return encoder_output, -jnp.inf  # Deterministic encoder returns z, we return (z, -jnp.inf) for consistency

    @nn.compact
    def decode(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """Decoder that maps latent z to output space."""
        return self.decoder(x, training)

    @nn.compact
    def dz_dt(self, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """Flow model that computes dz/dt using CRN with CT-style vector field."""
        # Optimized: minimize broadcasting and avoid redundant operations
        t = jnp.asarray(t)
        z, x, t = self._broadcast_inputs(z, x, t)

        target_estimate = self.crn_output(z, x, t, training=training)
        # Get alpha_t and gamma_prime directly from noise schedule
        t_expanded = jnp.expand_dims(jnp.asarray(t), axis=tuple(range(-self.z_ndims, 0)))
        alpha_t, gamma_prime_t = self.get_noise_params(t_expanded)
        return self.noise_schedule.lazy_flow_from_target(z, target_estimate, alpha_t, gamma_prime_t)
    

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray, key: jr.PRNGKey, training: bool = True) -> jnp.ndarray:
        # For initialization, we need to call the nn compact methods to initialize parameters

        batch_shape = y.shape[:-self.y_ndims]
                
        # Call flow_model to initialize its parameters (need dummy z and t)
        dummy_z = jnp.zeros(batch_shape + self.z_shape)
        dummy_t = jnp.zeros(batch_shape)
        
        # Call get_noise_params to initialize noise schedule parameters
        self.get_noise_params(dummy_t)
        
        # Call flow_model to initialize the CRN model parameters and alpha_gamma_prime_t
        dz_dt = self.dz_dt(dummy_z, x, dummy_t, training)
        
        # Call encoder and decoder to initialize their parameters
        # These @nn.compact methods need to be called during initialization to create parameters
        encoder_output = self.encode(y, training)
        decoder_output = self.decode(dummy_z, training)
        
        # For initialization, we just return a dummy output
        # The actual forward pass logic is handled by the individual methods
        return jnp.zeros(batch_shape + self.z_shape)

    @partial(jax.jit, static_argnums=(0, 5))
    def loss(self, params: dict, x: Optional[jnp.ndarray], y: jnp.ndarray, key: jr.PRNGKey, training: bool = True) -> Tuple[jnp.ndarray, dict]:
        """
        Compute the CT-style SNR-weighted loss.
        """
        # Extract config values at the start (self is static, so this is safe)
        # Convert to concrete Python types to avoid tracing issues
        use_snr_weight = bool(self.config.main.get('use_snr_weight', True))
        normalize_snr_weight = bool(self.config.main.get("normalize_snr_weight", False))
        recon_loss_type = str(self.config.main.get("recon_loss_type", "mse"))
        recon_weight = float(self.config.main.get("recon_weight", 0.0))
        reg_weight = float(self.config.main.get("reg_weight", 0.0))
        vae_weight = float(self.config.main.get("vae_weight", 1.0))
        
        # Split keys for random sampling operations (not dropout)
        key, t_key, noise_key, z_target_key = jr.split(key, 4)
        batch_shape = y.shape[:-self.y_ndims]

        # Encode Target (noisy latent)
        mu_z_target, logvar_z_target = self.apply(params, y, method='encode', training=training, rngs={'dropout': key})
        z_target = mu_z_target + jr.normal(z_target_key, mu_z_target.shape) * jnp.exp(0.5 * logvar_z_target)
        
        # Sample time and get noise schedule parameters
        t = jr.uniform(t_key, batch_shape, minval=0.0, maxval=1.0)
        t_expanded = jnp.expand_dims(t, axis=tuple(range(-self.z_ndims, 0)))     
        # Get noise schedule parameters (expand t to match expected shape)
        alpha_t, gamma_prime_t = self.apply(params, t_expanded, method='get_noise_params')
        
        sqrt_alpha_t = jnp.sqrt(alpha_t)
        sqrt_1_minus_alpha_t = jnp.sqrt(1.0 - alpha_t)
        
        # Compute noisy latent state at time t
        z_t = sqrt_alpha_t * z_target + sqrt_1_minus_alpha_t * jr.normal(noise_key, z_target.shape)
        
        # Compute Target estimate and flow prediction
        z_target_est = self.apply(params, z_t, x, t, method='crn_output', training=training, rngs={'dropout': key})
        dz_dt = self.apply(params, z_t, z_target_est, alpha_t, gamma_prime_t, method='lazy_flow_from_target')

        # Compute Predictions
        y_pred = self.apply(params, z_target_est, method='decode', training=training, rngs={'dropout': key})
        # VAE loss: decode z_target directly
        y_vae = self.apply(params, z_target, method='decode', training=training, rngs={'dropout': key})
        
        # Squeeze alpha_t and gamma_prime_t for use in SNR weight computation
        if use_snr_weight:
            snr_weight = self.apply(params, alpha_t, gamma_prime_t, method='lazy_target_snr')
            snr_weight = jnp.squeeze(snr_weight, axis=tuple(range(-self.z_ndims, 0)))
            # Normalize SNR weights by their mean if normalize_snr_weight is True
            if normalize_snr_weight:
                snr_weight_mean = jnp.mean(snr_weight)
                snr_weight = snr_weight / (snr_weight_mean + 1e-8)
        else: 
            snr_weight = 1.0

        # Compute Losses
        squared_error = jnp.mean((z_target_est - z_target) ** 2, axis=tuple(range(-self.z_ndims, 0)))
        flow_loss = jnp.mean(snr_weight * squared_error)
        reg_loss = jnp.mean(dz_dt**2)

        if recon_loss_type == "cross_entropy":
            recon_loss = optax.losses.safe_softmax_cross_entropy(y_pred, y)
            vae_loss   = optax.losses.safe_softmax_cross_entropy(y_vae, y)
        elif recon_loss_type == "mse":
            recon_loss = jnp.mean((y - y_pred)**2, axis=tuple(range(-self.y_ndims, 0)))
            vae_loss   = jnp.mean((y - y_vae)**2, axis=tuple(range(-self.y_ndims, 0)))
        recon_loss = jnp.mean(snr_weight * recon_loss) 
        vae_loss = jnp.mean(vae_loss)

        # # KL regularization term: KL(q(z_0|z_target), p(z_0))
        # # alpha_0 is guaranteed to be in (0, 1) by noise schedule clipping
        # # Get alpha_bar values at boundaries
        # alpha_0 = self.apply(params, jnp.asarray(1e-6), method='get_alpha_bar')
        # q_sigma_sq = 1.0 - alpha_0
        # mean_q_mu_sq_over_sigma_sq = alpha_0/q_sigma_sq*jnp.mean(jnp.sum(z_target**2, axis=tuple(range(-self.z_ndims, 0))))
        # kl_z0_loss = 0.5 * (mean_q_mu_sq_over_sigma_sq + self.z_dim*(jnp.log(q_sigma_sq) - 1.0))

        total_loss = flow_loss + recon_weight * recon_loss + reg_weight * reg_loss + vae_weight * vae_loss # + kl_z0_loss 
        
        return total_loss, {
            'flow_loss': flow_loss,
            'recon_loss': recon_loss, 
            'reg_loss': reg_loss,
            'vae_loss': vae_loss,
            'kl_z0_loss': 0.0, # kl_z0_loss,
            'total_loss': total_loss
        }

    
    @partial(jax.jit, static_argnums=(0, 3, 4, 5))  # self, num_steps, integration_method, output_type are static arguments
    def predict(self, params: dict, x: jnp.ndarray, num_steps: int = 20, integration_method: str = "midpoint", output_type: str = "end_point", prng_key: jr.PRNGKey = None) -> jnp.ndarray:
        """
        Generate predictions using ODE solver integration.
        
        Args:
            params: Model parameters
            x: Input data [batch_shape + x_shape]
            num_steps: Number of integration steps (default: 20)
            integration_method: Integration method ("euler", "midpoint", etc.) (default: "midpoint")
            output_type: Type of output ("end_point" for final prediction, "trajectory" for full trajectory)
            prng_key: Optional PRNG key for generative mode. If provided, samples z_0 from unit normal instead of zero.
            
        Returns:
            If output_type="end_point": Final prediction [batch_shape + y_shape]
            If output_type="trajectory": Full trajectory [num_steps, batch_shape + y_shape]
        """
        # Get latent trajectory using predict_latent
        z = self.predict_latent(params, x, num_steps, integration_method, output_type, prng_key)
        
        # Apply decoder to get final predictions
        return self.apply(params, z, method='decode', training=False)

    @partial(jax.jit, static_argnums=(0, 3, 4, 5))  # self, num_steps, integration_method, output_type are static arguments
    def predict_latent(self, params: dict, x: jnp.ndarray, num_steps: int = 20, integration_method: str = "midpoint", output_type: str = "end_point", prng_key: jr.PRNGKey = None) -> jnp.ndarray:
        """
        Make predictions using ODE solver integration, returning latent trajectories without decoding.
        
        This method is identical to predict() but skips the decoder step, returning the latent
        representation z directly. Useful for visualizing latent space trajectories.
        
        Args:
            params: Model parameters
            x: Input data [batch_shape + x_shape]
            num_steps: Number of integration steps (default: 20)
            integration_method: Integration method ("euler", "midpoint", etc.) (default: "midpoint")
            output_type: Type of output ("end_point" for final latent, "trajectory" for full trajectory)
            prng_key: Optional PRNG key for generative mode. If provided, samples z_0 from unit normal instead of zero.
            
        Returns:
            If output_type="end_point": Final latent state [batch_shape + z_shape]
            If output_type="trajectory": Full latent trajectory [num_steps, batch_shape + z_shape]
        """
        # Disable gradient tracking through parameters for inference
        params_no_grad = jax.lax.stop_gradient(params)
        batch_shape = x.shape[:-self.x_ndims]
        
        # Generate initial latent state z_0
        if prng_key is not None:
            # Generative mode: sample from unit normal distribution
            z_0 = jr.normal(prng_key, batch_shape + (self.z_dim,))
        else:
            # Regression mode: start from zero
            z_0 = jnp.zeros(batch_shape + (self.z_dim,))  # ode expects vectorized z
        
        # Define the vector field
        def vector_field(params, z, x, t):
            z = self._unflatten_z(z)
            dz_dt = self.apply(params, z, x, t, method='dz_dt', training=False)
            return self._flatten_z(dz_dt)
        
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
        return z  # Return latent directly without decoding

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
        return self.apply(params, z, method='decode', training=False)


    @partial(jax.jit, static_argnums=(0, 5, 7))  # self, optimizer, training are static
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


