import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
from flax.core import FrozenDict
import optax
from typing import Tuple, Dict, Optional
import inspect

from functools import partial, cached_property

# Import directly without going through src package to avoid einops dependency
from src.flow_models.config import Config
from src.vae.encoders import create_encoder
from src.vae.decoders import create_decoder
from src.flow_models.crns.crn import create_conditional_resnet
from src.embeddings.noise_schedules import create_noise_schedule
from src.utils.ode_integration import integrate_ode


class VAE_flow(nn.Module):
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

        self.no_noise_shedule = bool(self.config.main.get('no_noise_schedule', True))
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
        batch_shape = y.shape[:-self.y_ndims]
        # Call flow_model to initialize its parameters (need dummy z and t)
        dummy_z = jnp.zeros(batch_shape + self.z_shape)
        dummy_t = jnp.zeros(batch_shape)
        
        # Call get_noise_params to initialize noise schedule parameters
        if not self.no_noise_shedule:
            self.get_noise_params(dummy_t)
            self.get_alpha_bar(dummy_t)
        # initialize model components
        flow_output = self.dz_dt(dummy_z, x, dummy_t, training)
        encoder_output = self.encode(y, training)
        decoder_output = self.decode(dummy_z, training)
        
        return jnp.zeros(batch_shape + self.z_shape) # batch consistent with expectations

    @partial(jax.jit, static_argnums=(0, 5))    
    def loss(self, params: dict, x: Optional[jnp.ndarray], y: jnp.ndarray, key: jr.PRNGKey, training: bool = True, x_mask: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, dict]:
        """
        Compute the loss by calling individual @nn.compact methods with proper rngs.
        """
        # Extract config values at the start (self is static, so this is safe)
        # Convert to concrete Python types to avoid tracing issues
        recon_loss_type = str(self.config.main.get("recon_loss_type", "mse"))
        recon_weight = float(self.config.main.get("recon_weight", 0.0))
        reg_weight = float(self.config.main.get("reg_weight", 0.0))
        vae_weight = float(self.config.main.get("vae_weight", 0.0))
        use_snr_weight = bool(self.config.main.get('use_snr_weight', False))
        normalize_snr_weight = bool(self.config.main.get("normalize_snr_weight", True))
        
        # Split keys for random sampling operations (not dropout)
        key, t_key, z_0_key, z_t_noise_key, z_target_key, vae_noise_key = jr.split(key, 6)
        batch_shape = y.shape[:-self.y_ndims]

        # Encode Target (noisy latent)
        mu_z_target, logvar_z_target = self.apply(params, y, method='encode', training=training, rngs={'dropout': key})
        z_target = mu_z_target + jr.normal(z_target_key, mu_z_target.shape) * jnp.exp(0.5 * logvar_z_target)

        # Sample initial latent state and time
        t = jr.uniform(t_key, batch_shape, minval=0.0, maxval=1.0)
        t_expanded = jnp.expand_dims(t, axis=tuple(range(-self.z_ndims, 0)))
        z_0 = jr.normal(z_0_key, batch_shape + self.z_shape)
        
        # Sample latent state at time t using linear noise schedule
        diff_z = z_target - z_0
        z_t = z_0 + t_expanded * diff_z

        # Compute Flow Field and Target Estimate
        dz_dt = self.apply(params, z_t, x, t, x_mask, method='dz_dt', training=training, rngs={'dropout': key})    
        z_target_est = dz_dt * (1.0-t_expanded) + z_t   # assumes linear noise schedule

        # Compute Predictions
        y_pred = self.apply(params, z_target_est, method='decode', training=training, rngs={'dropout': key})
        y_vae = self.apply(params, z_target, method='decode', training=training, rngs={'dropout': key})

        # # Compute Losses (for linear noise schedule snr is simply alpha/alpha' = t)
        if use_snr_weight:
            snr_weight = t
            if normalize_snr_weight:
                snr_weight_mean = jnp.mean(snr_weight)
                snr_weight = snr_weight / (snr_weight_mean + 1e-8)
        else:
            snr_weight = 1.0
            # Normalize SNR weights by their mean if normalize_snr_weight is True

        squared_error = jnp.mean((dz_dt - diff_z)**2, axis=tuple(range(-self.z_ndims, 0)))
        flow_loss = jnp.mean(snr_weight * squared_error)
        reg_loss = jnp.mean(dz_dt**2)
        
        vae_loss = 0.0
        recon_loss = 0.0

        if use_snr_weight: 
            recon_snr = 1.0/(1-t)**2
            if normalize_snr_weight:
                recon_snr_mean = jnp.mean(recon_snr)
                recon_snr = recon_snr / (recon_snr_mean + 1e-8)
        else:
            recon_snr = 1.0
            # Normalize SNR weights by their mean if normalize_snr_weight is True

        if recon_loss_type == "cross_entropy":
            recon_loss = optax.losses.safe_softmax_cross_entropy(y_pred, y)
            vae_loss   = optax.losses.safe_softmax_cross_entropy(y_vae, y)
        elif recon_loss_type == "mse":
            recon_loss = jnp.mean((y - y_pred)**2, axis=tuple(range(-self.y_ndims, 0)))     
            vae_loss   = jnp.mean((y - y_vae)**2, axis=tuple(range(-self.y_ndims, 0)))
        
        recon_loss = jnp.mean(recon_snr * recon_loss)  # Average over batch dimension if needed      
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