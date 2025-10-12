"""
NoProp-FM: Flow Matching NoProp implementation.

This module implements the flow matching variant of NoProp.
The key idea is to model the denoising process as a flow that transforms
a base distribution to the target distribution.
"""

from typing import Any, Dict, Optional, Tuple, Callable
from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

# NoProp-FM doesn't use noise schedules
from .utils.ode_integration import integrate_ode
from .utils.jacobian_utils import trace_jacobian


class NoPropFM(nn.Module):
    """Flow Matching NoProp implementation.
    
    This class implements the flow matching variant where the denoising
    process is modeled as a flow that transforms a base distribution
    to the target distribution over continuous time.
    """
    
    target_dim: int  # Dimension of target z
    model: nn.Module
    num_timesteps: int = 20
    integration_method: str = "euler"  # "euler" or "heun"
    reg_weight: float = 0.0  # Hyperparameter from the paper
    sigma_t: float = 0.1  # Standard deviation of noise added to z_t
    
    @nn.compact
    def __call__(
        self, 
        z: jnp.ndarray, 
        x: jnp.ndarray, 
        t: jnp.ndarray
    ) -> jnp.ndarray:
        """Initialize all parameters by calling all @nn.compact methods once.
        
        This method is only called during init() to set up the parameter tree.
        During training, we use apply(params, method=...) to call specific methods.
        
        Args:
            z: Noisy target [batch_size, z_dim]
            x: Input data [batch_size, height, width, channels]
            t: Time step [batch_size]
            
        Returns:
            Model output [batch_size, z_dim]
        """
        # For flow matching, dz/dt is just the model output
        return self.model(z, x, t)
    
    def dz_dt(
        self,
        params: Dict[str, Any],
        z: jnp.ndarray,
        x: jnp.ndarray,
        t: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute the vector field dz/dt = f(z, x, t).
        
        For flow matching, the vector field is simply the model output.
        The model learns to predict the direction of flow from the current state
        to the target distribution.
                 
        Args:
            params: Model parameters
            z: Current state [batch_size, num_classes]
            x: Input data [batch_size, height, width, channels]
            t: Current time [batch_size]
            
        Returns:
            Vector field dz/dt [batch_size, num_classes]
        """
        # For flow matching, dz/dt is just the model output
        return self.apply(params, z, x, t)
    
    @partial(jax.jit, static_argnums=(0,))  # self is static
    def compute_loss(
        self,
        params: Dict[str, Any],
        x: jnp.ndarray,
        target: jnp.ndarray,
        key: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Compute the NoProp-FM loss.
        
        For flow matching, the loss is simply the MSE between model output and target:
        L_FM = E[||model_output - target||²]
        
        Args:
            params: Model parameters
            x: Input data [batch_size, height, width, channels]
            target: Clean target [batch_size, num_classes]
            key: Random key for sampling t and z_t
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Infer batch size from input tensor
        batch_shape = x.shape[:-1]
        
        # Split keys for all random operations
        key, t_key, z0_key, z_t_noise_key = jax.random.split(key, 4)
        
        # Sample random timesteps
        t = jax.random.uniform(t_key, batch_shape, minval=0.0, maxval=1.0)[...,None]
        
        # Sample z_t using linear interpolation: z_t = (1-t) * z0 + t * z_target + noise
        # where z0 ~ N(0, I) is sampled from unit normal distribution
        z_0 = jax.random.normal(z0_key, target.shape)

        z_t = (1.0 - t) * z_0 + t * target
        z_t = z_t + self.sigma_t * jax.random.normal(z_t_noise_key, target.shape)

        # Get model output
        dz_dt = self.apply(params, z_t, x, t)
        z_1_est = z_t + dz_dt*(1.0-t)

        # Compute MSE loss
        squared_error = (z_1_est - target) ** 2
        mse = jnp.mean(squared_error)
        
        # Regularization loss
        reg_loss = jnp.mean(dz_dt ** 2)
        no_prop_fm_loss = jnp.mean((dz_dt - (target - z_0)) ** 2)

        total_loss = no_prop_fm_loss + self.reg_weight * reg_loss

        # Compute additional metrics
        metrics = {
            "mse": mse,
            "reg_loss": reg_loss,
            "total_loss": total_loss,
            "no_prop_fm_loss": no_prop_fm_loss,
        }
        
        return total_loss, metrics
    
    @partial(jax.jit, static_argnums=(0, 3, 4, 5, 6))  # self, integration_method, output_dim, num_steps, output_type are static arguments    
    def predict(
        self,
        params: Dict[str, Any],
        x: jnp.ndarray,
        output_dim: int,
        num_steps: int,
        integration_method: str = "euler",
        output_type: str = "end_point"
    ) -> jnp.ndarray:
        """
        Generate predictions using the trained NoProp-FM neural ODE.
        
        This integrates the learned vector field from zeros (t=0)
        to the final prediction (t=1), following the paper's approach.
        Uses scan-based integration for better performance.
        
        Args:
            params: Model parameters
            x: Input data [batch_size, height, width, channels]
            integration_method: Integration method to use ("euler", "heun", "rk4")
            output_dim: Output dimension
            num_steps: Number of integration steps
            
        Returns:
            Final prediction [batch_size, output_dim]
        """
        # Disable gradient tracking through parameters for inference
        params_no_grad = jax.lax.stop_gradient(params)
        
        # Infer batch size from input tensor
        batch_shape = x.shape[:-1]
        
        # Start with zeros at t=0
        z0 = jnp.zeros(batch_shape + (output_dim,))
        
        # Create a static vector field function to avoid recompilation
        def vector_field(params, z, x, t):
            return self.dz_dt(params, z, x, t)
        
        # Use the main integrate_ode function which handles method dispatch
        z_final = integrate_ode(
            vector_field=vector_field,
            params=params_no_grad,
            z0=z0,
            x=x,
            time_span=(0.0, 1.0),
            num_steps=num_steps,
            method=integration_method,
            output_type=output_type
        )
        
        return z_final

    
    def predict_trajectory(
        self,
        params: Dict[str, Any],
        x: jnp.ndarray,
        integration_method: str,
        output_dim: int,
        num_steps: int
    ) -> jnp.ndarray:
        """Generate prediction trajectories using the trained NoProp-FM neural ODE.
        
        This is a wrapper around the predict method with output_type="trajectory".
        It integrates the learned vector field from zeros (t=0) to the final prediction (t=1),
        returning the full time course.
        
        Args:
            params: Model parameters
            x: Input data [batch_size, height, width, channels]
            integration_method: Integration method to use ("euler", "heun", "rk4")
            output_dim: Output dimension
            num_steps: Number of integration steps
            
        Returns:
            Full trajectory [num_steps + 1, batch_size, output_dim] (includes initial state)
        """
        return self.predict(
            params=params,
            x=x,
            output_dim=output_dim,
            num_steps=num_steps,
            integration_method=integration_method,
            output_type="trajectory"
        )
    
    def trace_jacobian(
        self,
        params: Dict[str, Any],
        z: jnp.ndarray,
        x: jnp.ndarray,
        t: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute the trace of the Jacobian of dz/dt with respect to z.
        
        This computes tr(∂f/∂z) where f(z,x,t) = dz/dt.
        This is useful for computing the divergence of the vector field,
        which is important for normalizing flows and log-determinant computations.
        
        Args:
            params: Model parameters
            z: Current state [batch_size, target_dim]
            x: Input data [batch_size, x_dim]
            t: Current time [batch_size]
            
        Returns:
            Trace of Jacobian [batch_size]
        """
        return trace_jacobian(self.dz_dt, params, z, x, t)
    
    @partial(jax.jit, static_argnums=(0, 6))  # self and optimizer are static arguments
    def train_step(
        self,
        params: Dict[str, Any],
        opt_state: optax.OptState,
        x: jnp.ndarray,
        target: jnp.ndarray,
        key: jax.random.PRNGKey,
        optimizer: optax.GradientTransformation
    ) -> Tuple[Dict[str, Any], optax.OptState, jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Single training step for NoProp-FM.
        
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
        
        # Update parameters using optimizer
        updates, updated_opt_state = optimizer.update(grads, opt_state, params)
        updated_params = optax.apply_updates(params, updates)
        
        return updated_params, updated_opt_state, loss, metrics