import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import flax.linen as nn
from flax.core import FrozenDict, freeze, unfreeze
import optax
from typing import Tuple, Dict, Optional
import inspect

from functools import partial, cached_property

# Import directly without going through src package to avoid einops dependency
from src.flow_models.config import Config
from src.vae.encoders import create_encoder
from src.vae.decoders import create_decoder
from src.flow_models.crn import create_conditional_resnet
from src.utils.ode_integration import integrate_ode
from src.flow_models.flow_planner import create_flow_planner


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
        
        self.flow_planner = create_flow_planner(
            self.config.flow_planner,
            latent_dim=self.z_dim
        )
        
        # Store config values as instance variables for use in JIT-compiled functions
        self.recon_loss_type = self.config.main.get("recon_loss_type", "mse")
        self.recon_weight = float(self.config.main.get("recon_weight", 0.0))
        self.reg_weight = float(self.config.main.get("reg_weight", 0.0))
        self.vae_weight = float(self.config.main.get("vae_weight", 0.0))
        # sample_method is stored in flow_planner instance (self.flow_planner.sample_method)
        
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

    @nn.compact
    def update_gmm_params(
        self,
        z_target_flat: jnp.ndarray,
        N_eff: float = 2000.0,
        lr: float = 0.2,
        training: bool = False
    ) -> dict:
        """
        Update GMM parameters using VBEM.
        
        This method wraps the flow_planner.gmm.update call so it can be called
        by method name (string) instead of using a lambda, allowing it to work
        in JIT-compiled functions.
        
        Args:
            z_target_flat: Flattened target latent vectors [N, z_dim]
            N_eff: Effective number of data points
            lr: Learning rate for VBEM updates
            training: Whether in training mode
            
        Returns:
            Updated GMM parameters dictionary
        """
        return self.flow_planner.gmm.update(z_target_flat, N_eff=N_eff, lr=lr, training=training)
    
    @nn.compact
    def sample_z_0(self, z_target: jnp.ndarray, key: jr.PRNGKey, training: bool = True) -> jnp.ndarray:
        """Sample initial latent state z_0 from target z_target using flow planner.
        
        Flattens z_target to vector format for flow planner, then unflattens the result.
        """
        # Flatten z_target to [batch, z_dim] for flow planner
        z_target_flat = self._flatten_z(z_target)
        
        # Sample from flow planner (expects [batch, latent_dim])
        z_0_flat = self.flow_planner.sample_x_0(z_target_flat, key, training=training)
        
        # Unflatten back to original shape
        z_0 = self._unflatten_z(z_0_flat)
        
        return z_0
    
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray, key: jr.PRNGKey, training: bool = True) -> jnp.ndarray:
        # For initialization, we need to call the nn compact methods to initialize parameters

        # Handle generative mode where x is None
        batch_shape = y.shape[:-self.y_ndims]
        # Call flow_model to initialize its parameters (need dummy z and t)
        dummy_z = jnp.zeros(batch_shape + self.z_shape)
        dummy_t = jnp.zeros(batch_shape)
        
        # initialize model components
        flow_output = self.dz_dt(dummy_z, x, dummy_t, training)
        encoder_output = self.encode(y, training)
        decoder_output = self.decode(dummy_z, training)
        
        # Initialize flow planner by calling sample_z_0 with dummy inputs
        # This ensures the flow planner's GMM and other components are initialized
        dummy_z_target = jnp.zeros(batch_shape + self.z_shape)
        dummy_key = jr.PRNGKey(0)
        _ = self.sample_z_0(dummy_z_target, dummy_key, training=training)
        
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
        
        # Split keys for random sampling operations (not dropout)
        key, t_key, z_0_key, z_0_noise_key, z_target_key, vae_noise_key = jr.split(key, 6)
        batch_shape = y.shape[:-self.y_ndims]

        # Encode Target (noisy latent)
        mu_z_target, logvar_z_target = self.apply(params, y, method='encode', training=training, rngs={'dropout': key})
        z_target = mu_z_target + jr.normal(z_target_key, mu_z_target.shape) * jnp.exp(0.5 * logvar_z_target)

        # Sample initial latent state using flow planner
        z_0 = self.apply(
            params,
            z_target,
            z_0_key,
            method='sample_z_0',
            training=training
        )
        
        # Compute GMM loss separately based on z_target
        z_target_flat = self._flatten_z(z_target)
        gmm_loss = self.apply(
            params,
            z_target_flat,
            method='compute_gmm_loss',
            training=training
        )

        
        # Sample time and compute linear interpolation
        t = jr.uniform(t_key, batch_shape, minval=0.0, maxval=1.0)
        t_expanded = jnp.expand_dims(t, axis=tuple(range(-self.z_ndims, 0)))
        
        # Linear interpolation: z_t = (1-t) * z_0 + t * z_target
        z_t = (1.0 - t_expanded) * z_0 + t_expanded * z_target
        
        # Compute flow direction (target is z_target - z_0)
        diff_z = z_target - z_0

        # Compute Flow Field
        dz_dt = self.apply(params, z_t, x, t, x_mask, method='dz_dt', training=training, rngs={'dropout': key})    
        
        # Target estimate from flow field
        z_target_est = z_t + (1.0 - t_expanded) * dz_dt

        # Compute Predictions
        y_pred = self.apply(params, z_target_est, method='decode', training=training, rngs={'dropout': key})
        
        # VAE loss: decode z_target directly
        y_vae = self.apply(params, z_target, method='decode', training=training, rngs={'dropout': key})

        # Compute Losses
        # Simple MSE loss for flow matching (no SNR weighting needed for linear paths)
        squared_error = jnp.mean((dz_dt - diff_z)**2, axis=tuple(range(-self.z_ndims, 0)))
        flow_loss = jnp.mean(squared_error)
        reg_loss = jnp.mean(dz_dt**2)

        if recon_loss_type == "cross_entropy":
            recon_loss = jnp.sum(-y * jnp.log(y_pred + 1e-8), axis = tuple(range(-self.y_ndims, 0)))
            vae_loss   = jnp.sum(-y * jnp.log(y_vae + 1e-8), axis = tuple(range(-self.y_ndims, 0)))
        elif recon_loss_type == "mse":
            recon_loss = jnp.sum((y - y_pred)**2, axis=tuple(range(-self.y_ndims, 0)))
            vae_loss   = jnp.sum((y - y_vae)**2, axis=tuple(range(-self.y_ndims, 0)))
        else:
            recon_loss = 0.0
            vae_loss = 0.0
        
        recon_loss = jnp.mean(recon_loss)  # Average over batch dimension
        vae_loss = jnp.mean(vae_loss)


        # KL regularization term: KL(q(z_0|z_target), p(z_0))
        # alpha_0 is guaranteed to be in (0, 1) by noise schedule clipping
        # q_sigma_sq = 1.0 - alpha_0
        # mean_q_mu_sq_over_sigma_sq = alpha_0/q_sigma_sq*jnp.mean(jnp.sum(z_target**2, axis=tuple(range(-self.z_ndims, 0))))
        # kl_z0_loss = 0.5 * (mean_q_mu_sq_over_sigma_sq + self.z_dim*(jnp.log(q_sigma_sq) - 1.0))

        # Get GMM loss weight from config (default to 0.0 if not specified)
        gmm_weight = float(self.config.main.get("gmm_weight", 0.0))

        total_loss = flow_loss + recon_weight * recon_loss + reg_weight * reg_loss + vae_weight * vae_loss + gmm_weight * gmm_loss # + kl_z0_loss  
        
        return total_loss, {
            'flow_loss': flow_loss,
            'recon_loss': recon_loss, 
            'reg_loss': reg_loss,
            'vae_loss': vae_loss,
            'gmm_loss': gmm_loss,
            'kl_z0_loss': 0.0, # kl_z0_loss,
            'total_loss': total_loss
        }


    @partial(jax.jit, static_argnums=(0, 3, 4, 5))  # self, num_steps, integration_method, output_type are static arguments
    def predict(self, params: dict, x: jnp.ndarray, num_steps: int = 20, integration_method: str = "euler", output_type: str = "end_point", prng_key: jr.PRNGKey = None) -> jnp.ndarray:
        """
        Make predictions using ODE solver integration.        
        Requires x is not None... use sample method for unconditional generation.
        """
        params_no_grad = jax.lax.stop_gradient(params)
        batch_shape = x.shape[:-self.x_ndims]
        
        if prng_key is not None:
            # Generative mode: sample z_0 unconditionally using flow planner
            # The flow itself is conditioned on x through the vector field
            key, sample_key = jr.split(prng_key, 2)
            
            # Create a dummy x_target for shape (needed by flow planner's sample_x_0)
            # Sinkhorn refinement won't be applied since training=False
            dummy_x_target = jnp.zeros(batch_shape + (self.z_dim,))
            
            # Use flow planner's sample_x_0 method (handles both mixture and normal cases)
            z_0 = self.apply(
                params,
                dummy_x_target,
                sample_key,
                method='sample_z_0',
                training=False
            )
        else:
            # Regression mode: start from zero
            z_0 = jnp.zeros(batch_shape + (self.z_dim,))  # ode expects vectorized z
        
        def vector_field(params, z, x, t):
            z = self._unflatten_z(z)  # crn expects unflattened z
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
        
        # Sample z_0 unconditionally using flow planner
        key, sample_key = jr.split(prng_key, 2)
        
        # Create a dummy x_target for shape (needed by flow planner's sample_x_0)
        # Sinkhorn refinement won't be applied since training=False
        dummy_x_target = jnp.zeros(batch_shape + (self.z_dim,))
        
        # Use flow planner's sample_x_0 method (handles both mixture and normal cases)
        z_0 = self.apply(
            params,
            dummy_x_target,
            sample_key,
            method='sample_z_0',
            training=False
        )
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
    

    @partial(jax.jit, static_argnums=(0, 5, 7, 8, 9, 10))  # self, optimizer, training, update_gmm, gmm_lr, N_eff are static arguments
    def train_step(
        self, 
        params: dict, 
        x: Optional[jnp.ndarray], 
        y: jnp.ndarray, 
        opt_state: dict, 
        optimizer, 
        key: jr.PRNGKey, 
        training: bool = True, 
        x_mask: Optional[jnp.ndarray] = None,
        update_gmm: bool = True,
        gmm_lr: float = 0.2,
        N_eff: float = 2000.0
    ) -> Tuple[dict, dict, jnp.ndarray, dict, dict]:
        """
        JIT-compiled training step for VAE with flow model and GMM flow planner.
        
        This method handles two types of parameter updates:
        1. GMM parameters: Updated via VBEM (not gradient descent) - returns updated params dict
        2. Flow model parameters (CRN, encoder, decoder): Updated via gradient descent
        
        Args:
            params: Current model parameters
            x: Conditional input [batch_size, input_dim] or [batch_size, seq_len, embed_dim] for sequences
            y: Target output [batch_size, output_dim] or [batch_size, seq_len, embed_dim] for sequences
            opt_state: Optimizer state
            optimizer: Optax optimizer
            key: Random key
            training: Whether in training mode
            x_mask: Boolean mask [batch_size, x_seq_len] for sequence models (True=valid, False=masked)
            update_gmm: Whether to update GMM parameters in this step
            gmm_lr: Learning rate for GMM VBEM updates (mixing parameter between 0 and 1)
            N_eff: Effective number of data points for GMM updates
            
        Returns:
            params: Updated model parameters (flow model only, GMM params unchanged)
            opt_state: Updated optimizer state
            loss: Training loss
            metrics: Training metrics
            updated_gmm_params: Updated GMM parameters dict (empty dict if update_gmm=False)
        """
        
        # Step 1: GMM parameter updates are handled outside train_step
        # because nested self.apply calls inside JIT-compiled functions can cause issues.
        # The update_gmm flag is kept for API compatibility.
        # Return empty dict - GMM updates should be computed in trainer before calling train_step
        updated_gmm_params = {}
        
        # Step 2: Compute loss and gradients for flow model parameters
        # GMM params are automatically excluded from gradients via extract_params() which applies stop_gradient
        def loss_fn(params):
            return self.loss(params, x, y, key, training=training, x_mask=x_mask)  
        
        (loss, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        
        # Step 3: Update flow model parameters using optimizer
        # GMM gradients are already zero because extract_params() uses stop_gradient
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss, metrics, updated_gmm_params
    
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