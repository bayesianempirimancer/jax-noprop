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
from src.embeddings.flow_schedules import create_flow_schedule
from src.utils.ode_integration import integrate_ode

class FlowModel(nn.Module):
    """Flow model using @nn.compact methods."""
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
        
        # Store config values as instance variables for use in JIT-compiled functions
        self.loss_type = self.config.main.get("loss_type", "flow_loss") # "flow, target, noise"
        self.normalize_snr_weight = bool(self.config.main.get("normalize_snr_weight", False))
        self.recon_loss_type = self.config.main.get("recon_loss_type", "mse")
        self.recon_weight = float(self.config.main.get("recon_weight", 0.0))
        self.reg_weight = float(self.config.main.get("reg_weight", 0.0))
        self.vae_weight = float(self.config.main.get("vae_weight", 0.0))

        # Initialize flow schedule
        # Get schedule type from flow_schedule config or fallback to main config
        schedule_config = dict(self.config.flow_schedule) if hasattr(self.config, 'flow_schedule') else {}
        
        if 'schedule_type' not in schedule_config:
             schedule_config['schedule_type'] = self.config.main.get("noise_schedule", "linear")

        # Ensure latent_shape is present (needed by create_flow_schedule)
        if 'latent_shape' not in schedule_config or schedule_config['latent_shape'] == "NA" or not schedule_config['latent_shape']:
             schedule_config['latent_shape'] = self.z_shape
             
        self.flow_schedule = create_flow_schedule(schedule_config)
        
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
    def sample_z_0(self, x_target: jnp.ndarray, key: jr.PRNGKey) -> jnp.ndarray:
        """Sample initial latent state from unit normal distribution."""
        return self.flow_schedule.sample_x_0(x_target, key)

    @nn.compact
    def z_t(self, z_0: jnp.ndarray, z_target: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Compute z_t using flow schedule."""
        return self.flow_schedule.x_t(z_0, z_target, t)

    @nn.compact
    def target_snr_weight(self, t: jnp.ndarray) -> jnp.ndarray:
        """Compute target SNR weight using flow schedule.  Always needed for recon loss."""
        weight = self.flow_schedule.target_snr_weight(t)
        return jnp.clip(weight, a_min=0.1, a_max=10.0)
    
    @nn.compact
    def snr_weight(self, t: jnp.ndarray) -> jnp.ndarray:
        """Compute noise SNR weight using flow schedule.  Depends on loss type."""

        if self.loss_type == "flow_loss":
            weight = self.flow_schedule.flow_snr_weight(t)
        elif self.loss_type == "target_loss":
            weight = self.flow_schedule.target_snr_weight(t)
        elif self.loss_type == "noise_loss":
            weight = self.flow_schedule.noise_snr_weight(t)
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}.  Must be one of 'flow_loss', 'target_loss', or 'noise_loss'.")
        
        return jnp.clip(weight, a_min=0.1, a_max=10.0)
            
    @nn.compact
    def dz_dt(self, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray, x_mask: Optional[jnp.ndarray] = None, training: bool = True) -> jnp.ndarray:
        """Flow model that computes dz/dt using CRN."""
        # Optimized: minimize broadcasting and avoid redundant operations
        t = jnp.asarray(t)
        z, x, t = self._broadcast_inputs(z, x, t)
        
        # For seq to seq models we may wish to encode x in the same manner as y before passing to seq_crn
        if self.config.main.get("encode_x", False) and x is not None:
            x = self.encode(x, training)[0]
        
        call_sig = inspect.signature(self.crn_model.__call__)
        if 'x_mask' in call_sig.parameters:
            crn_output = self.crn_model(z, x, t, x_mask=x_mask, training=training)            
        else:
            crn_output = self.crn_model(z, x, t, training=training)            

        if self.loss_type == "flow_loss":
            dz_dt = crn_output
        elif self.loss_type == "target_loss":
            dz_dt = self.flow_schedule.flow_from_target(z, crn_output, t)
        elif self.loss_type == "noise_loss":
            dz_dt = self.flow_schedule.flow_from_noise(z, crn_output, t)
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")
        return dz_dt
        
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
        
        # Call flow_schedule to initialize its parameters
        self.flow_schedule(dummy_z, dummy_z, dummy_t)
        # initialize model components
        flow_output = self.dz_dt(dummy_z, x, dummy_t, training)
        encoder_output = self.encode(y, training)
        decoder_output = self.decode(dummy_z, training)
        
        return jnp.zeros(batch_shape + self.z_shape) # batch consistent with expectations

    def compute_loss(self, x: Optional[jnp.ndarray], y: jnp.ndarray, key: jr.PRNGKey, training: bool = True, x_mask: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, dict]:
        """
        Compute the loss (runs inside apply).
        """
        # Extract config values
        recon_loss_type = str(self.config.main.get("recon_loss_type", "mse"))
        use_recon_snr_weight = bool(self.config.main.get('use_recon_snr_weight', False))
        recon_weight = float(self.config.main.get("recon_weight", 0.0))
        reg_weight = float(self.config.main.get("reg_weight", 0.0))
        vae_weight = float(self.config.main.get("vae_weight", 0.0))
        use_snr_weight = bool(self.config.main.get('use_snr_weight', False))
        normalize_snr_weight = bool(self.config.main.get("normalize_snr_weight", True))
        
        # Split keys
        key, t_key, z_0_key, z_t_noise_key, z_target_key, vae_noise_key = jr.split(key, 6)
        batch_shape = y.shape[:-self.y_ndims]

        # Encode Target (noisy latent)
        mu_z_target, logvar_z_target = self.encode(y, training=training)
        z_target = mu_z_target + jr.normal(z_target_key, mu_z_target.shape) * jnp.exp(0.5 * logvar_z_target)

        # Sample initial latent state and time
        t = jr.uniform(t_key, batch_shape, minval=0.0, maxval=1.0)
        z_0 = self.sample_z_0(z_target, key=z_0_key)
        z_t = self.z_t(z_0, z_target, t)
        
        # Broadcast inputs for CRN
        z_t_b, x_b, t_b = self._broadcast_inputs(z_t, x, t)
        
        # Encode x if needed
        if self.config.main.get("encode_x", False) and x_b is not None:
            x_b = self.encode(x_b, training)[0]
            
        # Get CRN output
        call_sig = inspect.signature(self.crn_model.__call__)
        if 'x_mask' in call_sig.parameters:
            crn_out = self.crn_model(z_t_b, x_b, t_b, x_mask=x_mask, training=training)            
        else:
            crn_out = self.crn_model(z_t_b, x_b, t_b, training=training)

        if self.loss_type == "flow_loss":
            dz_dt = crn_out
            z_target_est = self.flow_schedule.target_from_flow(z_t, dz_dt, t)
            dz_dt_target = self.flow_schedule.flow_from_endpoints(z_0, z_target, t)
            flow_loss = jnp.mean((dz_dt - dz_dt_target)**2, axis=tuple(range(-self.z_ndims, 0)))

        elif self.loss_type == "target_loss":
            z_target_est = crn_out
            dz_dt = self.flow_schedule.flow_from_target(z_t, z_target_est, t)
            flow_loss = jnp.mean((z_target_est - z_target)**2, axis=tuple(range(-self.z_ndims, 0)))

        elif self.loss_type == "noise_loss":
            noise_est = crn_out
            z_target_est = self.flow_schedule.target_from_noise(z_t, noise_est, t)
            dz_dt = self.flow_schedule.flow_from_noise(z_t, noise_est, t)
            flow_loss = jnp.mean((noise_est - z_0)**2, axis=tuple(range(-self.z_ndims, 0)))

        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}.  Must be one of 'flow_loss', 'target_loss', or 'noise_loss'.")

        # Compute Predictions
        y_pred = self.decode(z_target_est, training=training)
        y_vae = self.decode(z_target, training=training)

        # # Compute Losses
        if use_snr_weight:
            snr_weight = self.snr_weight(t)
            if normalize_snr_weight:
                snr_weight_mean = jnp.mean(snr_weight)
                snr_weight = snr_weight / (snr_weight_mean + 1e-8)
            flow_loss = jnp.mean(snr_weight * flow_loss)
        else:
            flow_loss = jnp.mean(flow_loss)

        reg_loss = jnp.mean(dz_dt**2)
        
        if use_recon_snr_weight: 
            recon_snr = self.flow_schedule.alpha(t)**2
            if normalize_snr_weight:
                recon_snr_mean = jnp.mean(recon_snr)
                recon_snr = recon_snr / (recon_snr_mean + 1e-8)
        else:
            recon_snr = 1.0

        if recon_loss_type == "cross_entropy":
            recon_loss = optax.losses.safe_softmax_cross_entropy(y_pred, y)
            vae_loss   = optax.losses.safe_softmax_cross_entropy(y_vae, y)
        elif recon_loss_type == "mse":
            recon_loss = jnp.mean((y - y_pred)**2, axis=tuple(range(-self.y_ndims, 0)))     
            vae_loss   = jnp.mean((y - y_vae)**2, axis=tuple(range(-self.y_ndims, 0)))
        
        recon_loss = jnp.mean(recon_snr * recon_loss)
        vae_loss = jnp.mean(vae_loss)

        total_loss = flow_loss + recon_weight * recon_loss + reg_weight * reg_loss + vae_weight * vae_loss
        
        return total_loss, {
            'flow_loss': flow_loss,
            'recon_loss': recon_loss, 
            'reg_loss': reg_loss,
            'vae_loss': vae_loss,
            'kl_z0_loss': 0.0,
            'total_loss': total_loss
        }

    @partial(jax.jit, static_argnums=(0, 5))    
    def loss(self, params: dict, x: Optional[jnp.ndarray], y: jnp.ndarray, key: jr.PRNGKey, training: bool = True, x_mask: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, dict]:
        """
        Compute the loss by calling compute_loss via apply.
        """
        return self.apply(params, x, y, key, training, x_mask, method=self.compute_loss, rngs={'dropout': key})

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