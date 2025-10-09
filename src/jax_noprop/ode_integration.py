"""
ODE integration methods for NoProp continuous-time variants.

This module provides various numerical integration methods for solving
neural ordinary differential equations (ODEs) used in NoProp-CT and NoProp-FM.
"""

from typing import Any, Callable, Dict, Tuple, Optional
import jax
import jax.numpy as jnp


def euler_step(
    vector_field: Callable,
    params: Dict[str, Any],
    z: jnp.ndarray,
    x: jnp.ndarray,
    t: jnp.ndarray,
    dt: float
) -> jnp.ndarray:
    """Single Euler integration step.
    
    This implements the forward Euler method:
    z_{t+dt} = z_t + dt * f(z_t, x, t)
    
    Args:
        vector_field: Function that computes dz/dt = f(z, x, t)
        params: Model parameters
        z: Current state [batch_size, state_dim]
        x: Input data [batch_size, ...]
        t: Current time [batch_size]
        dt: Time step size
        
    Returns:
        Updated state [batch_size, state_dim]
    """
    dz_dt = vector_field(params, z, x, t)
    return z + dt * dz_dt


def heun_step(
    vector_field: Callable,
    params: Dict[str, Any],
    z: jnp.ndarray,
    x: jnp.ndarray,
    t: jnp.ndarray,
    dt: float
) -> jnp.ndarray:
    """Single Heun integration step (2nd order Runge-Kutta).
    
    This implements the Heun method (improved Euler):
    1. k1 = f(z_t, x, t)
    2. k2 = f(z_t + dt*k1, x, t + dt)
    3. z_{t+dt} = z_t + dt/2 * (k1 + k2)
    
    Args:
        vector_field: Function that computes dz/dt = f(z, x, t)
        params: Model parameters
        z: Current state [batch_size, state_dim]
        x: Input data [batch_size, ...]
        t: Current time [batch_size]
        dt: Time step size
        
    Returns:
        Updated state [batch_size, state_dim]
    """
    # First stage
    k1 = vector_field(params, z, x, t)
    
    # Second stage
    z_pred = z + dt * k1
    t_next = t + dt
    k2 = vector_field(params, z_pred, x, t_next)
    
    # Combine stages
    return z + dt * 0.5 * (k1 + k2)


def rk4_step(
    vector_field: Callable,
    params: Dict[str, Any],
    z: jnp.ndarray,
    x: jnp.ndarray,
    t: jnp.ndarray,
    dt: float
) -> jnp.ndarray:
    """Single 4th order Runge-Kutta integration step.
    
    This implements the classic RK4 method:
    1. k1 = f(z_t, x, t)
    2. k2 = f(z_t + dt/2*k1, x, t + dt/2)
    3. k3 = f(z_t + dt/2*k2, x, t + dt/2)
    4. k4 = f(z_t + dt*k3, x, t + dt)
    5. z_{t+dt} = z_t + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    Args:
        vector_field: Function that computes dz/dt = f(z, x, t)
        params: Model parameters
        z: Current state [batch_size, state_dim]
        x: Input data [batch_size, ...]
        t: Current time [batch_size]
        dt: Time step size
        
    Returns:
        Updated state [batch_size, state_dim]
    """
    # Stage 1
    k1 = vector_field(params, z, x, t)
    
    # Stage 2
    z2 = z + dt * 0.5 * k1
    t2 = t + dt * 0.5
    k2 = vector_field(params, z2, x, t2)
    
    # Stage 3
    z3 = z + dt * 0.5 * k2
    k3 = vector_field(params, z3, x, t2)
    
    # Stage 4
    z4 = z + dt * k3
    t4 = t + dt
    k4 = vector_field(params, z4, x, t4)
    
    # Combine stages
    return z + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0


def adaptive_step(
    vector_field: Callable,
    params: Dict[str, Any],
    z: jnp.ndarray,
    x: jnp.ndarray,
    t: jnp.ndarray,
    dt: float,
    tolerance: float = 1e-6
) -> Tuple[jnp.ndarray, float]:
    """Adaptive step size integration.
    
    This uses error estimation to adaptively choose step sizes.
    It compares a full step with two half steps to estimate error.
    
    Args:
        vector_field: Function that computes dz/dt = f(z, x, t)
        params: Model parameters
        z: Current state [batch_size, state_dim]
        x: Input data [batch_size, ...]
        t: Current time [batch_size]
        dt: Initial time step size
        tolerance: Error tolerance for adaptive stepping
        
    Returns:
        Tuple of (updated_state, next_step_size)
    """
    # Full step
    z_full = heun_step(vector_field, params, z, x, t, dt)
    
    # Two half steps
    z_half1 = heun_step(vector_field, params, z, x, t, dt/2)
    z_half2 = heun_step(vector_field, params, z_half1, x, t + dt/2, dt/2)
    
    # Estimate error
    error = jnp.mean(jnp.abs(z_full - z_half2))
    
    # Adjust step size based on error
    if error > tolerance:
        # Reduce step size
        new_dt = dt * 0.5
    elif error < tolerance * 0.1:
        # Increase step size
        new_dt = dt * 1.5
    else:
        # Keep current step size
        new_dt = dt
    
    # Use the more accurate result (two half steps)
    return z_half2, new_dt


def integrate_ode(
    vector_field: Callable,
    params: Dict[str, Any],
    z0: jnp.ndarray,
    x: jnp.ndarray,
    time_span: Tuple[float, float],
    num_steps: int,
    method: str = "euler"
) -> jnp.ndarray:
    """Integrate an ODE using the specified method.
    
    This function integrates the ODE dz/dt = f(z, x, t) from t_start to t_end
    using the specified numerical method.
    
    Args:
        vector_field: Function that computes dz/dt = f(z, x, t)
        params: Model parameters
        z0: Initial state [batch_size, state_dim]
        x: Input data [batch_size, ...]
        time_span: Tuple of (start_time, end_time)
        num_steps: Number of integration steps
        method: Integration method ("euler", "heun", "rk4", "adaptive")
        
    Returns:
        Final state [batch_size, state_dim]
    """
    t_start, t_end = time_span
    dt = (t_end - t_start) / num_steps
    
    # Initialize
    z = z0
    t = jnp.full((z0.shape[0],), t_start)
    
    # Choose integration method
    if method == "euler":
        step_func = euler_step
    elif method == "heun":
        step_func = heun_step
    elif method == "rk4":
        step_func = rk4_step
    elif method == "adaptive":
        # For adaptive method, we need special handling
        return _integrate_adaptive(vector_field, params, z0, x, time_span, num_steps)
    else:
        raise ValueError(f"Unknown integration method: {method}")
    
    # Integrate
    for i in range(num_steps):
        z = step_func(vector_field, params, z, x, t, dt)
        t = t + dt
    
    return z


def _integrate_adaptive(
    vector_field: Callable,
    params: Dict[str, Any],
    z0: jnp.ndarray,
    x: jnp.ndarray,
    time_span: Tuple[float, float],
    max_steps: int
) -> jnp.ndarray:
    """Internal function for adaptive integration."""
    t_start, t_end = time_span
    dt = (t_end - t_start) / max_steps
    
    z = z0
    t = jnp.full((z0.shape[0],), t_start)
    current_dt = dt
    
    for i in range(max_steps):
        if jnp.all(t >= t_end):
            break
            
        # Ensure we don't overshoot
        remaining_time = t_end - t
        step_dt = jnp.minimum(current_dt, remaining_time)
        
        z, current_dt = adaptive_step(vector_field, params, z, x, t, step_dt)
        t = t + step_dt
    
    return z


def integrate_flow(
    vector_field: Callable,
    params: Dict[str, Any],
    z0: jnp.ndarray,
    x: jnp.ndarray,
    time_span: Tuple[float, float],
    num_steps: int,
    method: str = "euler"
) -> jnp.ndarray:
    """Integrate a flow (same as integrate_ode, for compatibility).
    
    This is an alias for integrate_ode to maintain compatibility
    with flow matching terminology.
    
    Args:
        vector_field: Function that computes dz/dt = f(z, x, t)
        params: Model parameters
        z0: Initial state [batch_size, state_dim]
        x: Input data [batch_size, ...]
        time_span: Tuple of (start_time, end_time)
        num_steps: Number of integration steps
        method: Integration method ("euler", "heun", "rk4", "adaptive")
        
    Returns:
        Final state [batch_size, state_dim]
    """
    return integrate_ode(vector_field, params, z0, x, time_span, num_steps, method)


# Default integration configurations
DEFAULT_INTEGRATION_METHODS = {
    "training": "euler",      # Fast for training
    "evaluation": "heun",     # More accurate for evaluation
    "high_precision": "rk4",  # High precision when needed
}

DEFAULT_NUM_STEPS = {
    "training": 20,
    "evaluation": 40,
    "high_precision": 100,
}
