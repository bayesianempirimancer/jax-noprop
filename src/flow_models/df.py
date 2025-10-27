"""
NoProp-CT: Continuous-time NoProp implementation.

This module implements the continuous-time variant of NoProp using neural ODEs.
The key idea is to model the denoising process as a continuous-time dynamical system.
"""

from typing import Any, Dict, Tuple, Optional
from functools import partial, cached_property
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
import optax

from src.configs.base_model import BaseModel
from src.configs.base_config import BaseConfig
from src.embeddings.noise_schedules import NoiseSchedule, LinearNoiseSchedule, CosineNoiseSchedule, SigmoidNoiseSchedule
from src.embeddings.noise_schedules import SimpleLearnableNoiseSchedule, LearnableNoiseSchedule
from src.utils.ode_integration import integrate_ode
from src.utils.jacobian_utils import trace_jacobian
from .crn import create_cond_resnet


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

@dataclass(frozen=True)
class Config(BaseConfig):
    """Configuration for NoProp DF Network."""
    
    # Set model_name from config_dict
    model_name: str = "noprop_df_net"
    
    # Properties for easy access
    
    # Hierarchical configuration structure
    config_dict = {
        "model_name": "noprop_df_net",
        "loss_type": "mse",
        
        # NoProp specific parameters (note that noise schedule is fixed for DF)
        "noise_schedule": "linear",
        "num_timesteps": 20,
        "integration_method": "euler",
        "reg_weight": 0.0,
        
        "model_config": {
            "output_dim": None,
            "hidden_dims": (128, 128, 128),
            "time_embed_dim": 64,
            "time_embed_method": "sinusoidal",
            "dropout_rate": 0.1,
            "activation_fn": "swish",
            "use_batch_norm": False,
        }
    }
    



# ============================================================================
# MODEL IMPLEMENTATION
# ============================================================================

class NoPropDF(BaseModel[Config]):
    """Diffusion NoProp implementation.
    
    This class implements a standard diffusion model where the neural network
    predicts noise instead of directly predicting the target or flow derivative.
    """
    
    config: Config
    z_shape: Tuple[int, ...]  # Shape of target z (excluding batch dimensions)
    x_ndims: int = 1  # Number of dimensions in input x
    model: str = "conditional_resnet_mlp"  # Model type string
    model_config: Optional[Config] = None
    noise_schedule: NoiseSchedule = SimpleLearnableNoiseSchedule()
    
    @property
    def z_ndims(self) -> int:
        """Number of dimensions in z_shape."""
        return len(self.z_shape)
        
    @property
    def z_dim(self) -> int:
        """Total flattened dimension of z."""
        return self._get_z_dim
    
    
    @cached_property
    def _get_z_dim(self) -> int:
        """Calculate the flattened dimension from z_shape."""
        z_dim = 1
        for dim in self.z_shape:
            z_dim *= dim
        return z_dim

    def _flatten_z(self, z: jnp.ndarray) -> jnp.ndarray:
        """Flatten the z tensor."""
        if len(self.z_shape) > 1:
            return z.reshape(z.shape[:-1] + (self._get_z_dim(),))
        else:
            return z

    def _unflatten_z(self, z: jnp.ndarray) -> jnp.ndarray:
        """Unflatten the z tensor."""
        if len(self.z_shape) > 1:
            return z.reshape(z.shape[:-1] + self.z_shape)
        else:
            return z

    @nn.compact
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray, training: bool = True, rngs: dict = None, **kwargs) -> jnp.ndarray:
        """
        Main forward pass to compute dz_dt
        
        Args:
            z: Current state/trajectory of shape (batch_size, z_dim)
            x: Input data of shape (batch_size, x_dim)
            t: Time step of shape (batch_size,)
            training: Whether in training mode (affects dropout)
            rngs: Random number generator keys
            **kwargs: Additional arguments (for HF compatibility)
            
        Returns:
            dz/dt [batch_shape + z_shape]
        """
        z = self._unflatten_z(z)
        
        # For diffusion, the neural network predicts noise
        predicted_noise = self._get_model_output(z, x, t, training=training)
        return predicted_noise

    @nn.compact
    def _get_model_output(
        self, 
        z: jnp.ndarray, 
        x: jnp.ndarray, 
        t: jnp.ndarray,
        training: bool = True
    ) -> jnp.ndarray:
        """Get model output - @nn.compact method for parameter initialization."""
        from .crn import create_flow_model
        
        # Use the convenience function to handle all model creation logic
        return create_flow_model(self.model, z, x, t, training)

    @nn.compact
    def get_gamma_gamma_prime_t(self, t: jnp.ndarray):
        """Get noise schedule output using @nn.compact method."""
        return self.noise_schedule(t)
    
    def dz_dt(
        self,
        params: Dict[str, Any],
        z: jnp.ndarray,
        x: jnp.ndarray,
        t: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute dz/dt for the diffusion process.
        
        This is just a wrapper for the __call__ method.
        
        Args:
            params: Model parameters
            z: Current state [batch_size + z_shape]
            x: Input data [batch_size, ...]
            t: Current time [batch_size]
            
        Returns:
            Vector field dz/dt [batch_size + z_shape]
        """

        predicted_noise = self.apply(params, z, x, t, training=False)
        dz_dt = 0.5*z - predicted_noise 
        return dz_dt
    
    @partial(jax.jit, static_argnums=(0,))  # self is static
    def compute_loss(
        self,
        params: Dict[str, Any],
        x: jnp.ndarray,
        target: jnp.ndarray,
        key: jr.PRNGKey
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Compute the diffusion loss.
        
        For diffusion, the loss is MSE between predicted noise and actual noise:
        L_diff = E[||model_output - noise||²]
        
        Args:
            params: Model parameters
            x: Input data [batch_shape, ...]
            target: Clean target [batch_shape + z_shape]
            key: Random key for sampling t and noise
            
        Returns:
            Tuple of (loss, metrics)
        """
        target = self._flatten_z(target)
        batch_shape = target.shape[:-1]
        
        # Split keys for all random operations
        key, t_key, noise_key = jr.split(key, 3)
        
        # Sample random timesteps
        t = jr.uniform(t_key, batch_shape, minval=0.0, maxval=1.0)
        
        # Get noise schedule (linear: alpha_t = t)
        alpha_t = t
        alpha_t_sqrt = jnp.sqrt(alpha_t)
        one_minus_alpha_t_sqrt = jnp.sqrt(1.0 - alpha_t)
        
        # Sample noise
        noise = jr.normal(noise_key, target.shape)
        
        # Create noisy target: z_t = sqrt(alpha_t) * target + sqrt(1-alpha_t) * noise
        z_t = alpha_t_sqrt[..., None] * target + one_minus_alpha_t_sqrt[..., None] * noise
        
        # Get model output (predicted noise)
        key, dropout_key = jr.split(key)
        predicted_noise = self.apply(params, z_t, x, t, rngs={'dropout': dropout_key})
        
        # Compute MSE loss between predicted and actual noise
        squared_error = (predicted_noise*one_minus_alpha_t_sqrt[..., None] - noise) ** 2
        

        # Regularization loss
        reg_loss = jnp.mean(predicted_noise ** 2)        
        if self.config.loss_type == "snr_weighted_mse":
            mse = None
            snr_weight = 1.0/(1.0-alpha_t)
            snr_weight_mean = jnp.mean(snr_weight)
            snr_weight = snr_weight / snr_weight_mean
            snr_weighted_loss = jnp.mean(snr_weight[..., None] * squared_error)
            total_loss = snr_weighted_loss + self.config.reg_weight * reg_loss
        else:
            mse = jnp.mean(squared_error)
            total_loss = mse + self.config.reg_weight * reg_loss
            snr_weighted_loss = None
            snr_weight_mean = None
        
        # Compute additional metrics
        metrics = {
            "mse": mse,
            "reg_loss": reg_loss,
            "total_loss": total_loss,
            "snr_weighted_loss": snr_weighted_loss,
            "snr_weight_mean": snr_weight_mean,
        }
        
        return total_loss, metrics
        
    @partial(jax.jit, static_argnums=(0, 3, 4, 5, 6))  # self, num_steps, integration_method, output_type, with_logp are static arguments    
    def predict(
        self,
        params: Dict[str, Any],
        x: jnp.ndarray,
        num_steps: int,
        integration_method: str = "euler",
        output_type: str = "end_point",
        with_logp: bool = False,
        key: jr.PRNGKey = None
    ) -> jnp.ndarray:
        """
        Generate predictions using the trained NoProp-CT neural ODE.
        
        This integrates the learned vector field from zeros (t=0)
        to the final prediction (t=1), following the paper's approach.
        Uses scan-based integration for better performance.
        
        Args:
            params: Model parameters
            x: Input data [batch_shape, ...]
            num_steps: Number of integration steps
            integration_method: Integration method to use ("euler", "heun", "rk4", "adaptive")
            output_type: Type of output ("end_point" or "trajectory")
            key: Random key
            
        Returns:
            If output_type="end_point": Final prediction [batch_shape + z_shape]
            If output_type="trajectory": Full trajectory [num_steps+1, batch_shape + z_shape]
            Note that in fully generative model you need to provide a key, and if x=None, then you need
                to provide something that returns x.shape[:-self.x_ndims] = (number_of_samples,)
        """
        # Disable gradient tracking through parameters for inference
        params_no_grad = jax.lax.stop_gradient(params)
        
        # Infer batch shape from input tensor
        batch_shape = x.shape[:-self.x_ndims]

        # For diffusion, always start from pure noise
        z0 = jnp.zeros(batch_shape + self.z_shape)
        if key is not None:
            z0 = z0 + jr.normal(key, batch_shape + self.z_shape)
        
        if with_logp:
            # Combined vector field for z and logp integration
            def vector_field(params, state, x, t):
                z = state[..., :-1]  # All but last dimension
                logp = state[..., -1:]  # Last dimension
                dz_dt = self.dz_dt(params, z, x, t)
                # Compute Jacobian trace for logp evolution
                # Broadcast t to match batch size
                trace_jac = trace_jacobian(self.dz_dt, params, z, x, t)
                dlogp_dt = -trace_jac
                return jnp.concatenate([dz_dt, dlogp_dt[..., None]], axis=-1)
            
            # Initialize combined state
            logp0 = jnp.zeros(batch_shape)
            state0 = jnp.concatenate([z0, logp0[..., None]], axis=-1)
            
            z_1 = integrate_ode(
                    vector_field=vector_field,
                    params=params_no_grad,
                    z0=state0,
                    x=x,
                    time_span=(0.0, 1.0),
                    num_steps=num_steps,
                    method=integration_method,
                    output_type=output_type
                )
            # For diffusion, no learned offset needed
        else:
            # Standard vector field for z only
            def vector_field(params, z, x, t):
                return self.dz_dt(params, z, x, t)

            z_1 = integrate_ode(
                    vector_field=vector_field,
                    params=params_no_grad,
                    z0=z0,
                    x=x,
                    time_span=(0.0, 1.0),
                    num_steps=num_steps,
                    method=integration_method,
                    output_type=output_type
                )

        return z_1



    @partial(jax.jit, static_argnums=(0, 5))  # self and optimizer are static arguments
    def train_step(
        self,
        params: Dict[str, Any],
        x: jnp.ndarray,
        target: jnp.ndarray,
        opt_state: optax.OptState,
        optimizer: optax.GradientTransformation,
        key: jr.PRNGKey,
    ) -> Tuple[Dict[str, Any], optax.OptState, jnp.ndarray, Dict[str, jnp.ndarray]]:

        """Single training step for NoProp-CT.
        
        Args:
            params: Model parameters
            opt_state: Optimizer state
            x: Input data [batch_size, height, width, channels]
            target: Clean target [batch_size, num_classes]
            key: Random key
            optimizer: Optax optimizer
            
        Returns:
            Tuple of (updated_params, updated_opt_state, loss, metrics)
    """
        # Compute loss and gradients (t and z_t are sampled inside compute_loss)
        # compute_loss is already JIT-compiled, so this will be fast
        (loss, metrics), grads = jax.value_and_grad(
            self.compute_loss, has_aux=True)(params, x, target, key)
        
        # Update parameters using optimizer (outside JIT)
        updates, updated_opt_state = optimizer.update(grads, opt_state, params)
        updated_params = optax.apply_updates(params, updates)
        
        return updated_params, updated_opt_state, loss, metrics
    