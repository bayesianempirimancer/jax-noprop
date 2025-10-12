"""
ODE integration methods for NoProp continuous-time variants.

This module provides various numerical integration methods for solving
neural ordinary differential equations (ODEs) used in NoProp-CT and NoProp-FM.
"""

from typing import Any, Callable, Dict, Tuple, Optional
import jax
import jax.numpy as jnp


# =============================================================================
# MAIN INTEGRATION FUNCTION AND DEFAULTS
# =============================================================================

def integrate_ode(
    vector_field: Callable,
    params: Dict[str, Any],
    z0: jnp.ndarray,
    x: jnp.ndarray,
    time_span: Tuple[float, float],
    num_steps: int,
    method: str = "euler",
    output_type: str = "end_point"
) -> jnp.ndarray:
    """Integrate an ODE using the specified method.
    
    This function integrates the ODE dz/dt = f(z, x, t) from t_start to t_end
    using the specified numerical method with scan-based implementation for
    better JIT compilation.
    
    Args:
        vector_field: Function that computes dz/dt = f(z, x, t)
        params: Model parameters
        z0: Initial state [batch_size, state_dim]
        x: Input data [batch_size, ...]
        time_span: Tuple of (start_time, end_time)
        num_steps: Number of integration steps
        method: Integration method ("euler", "heun", "rk4", "adaptive")
        output_type: Type of output ("end_point" or "trajectory")
        
    Returns:
        If output_type="end_point": Final state [batch_size, state_dim]
        If output_type="trajectory": Full trajectory [num_steps+1, batch_size, state_dim]
    """
    # Use scan-based JIT-compiled integration functions for better performance
    if output_type == "end_point":
        if method == "euler":
            return _integrate_ode_euler_scan(vector_field, params, z0, x, time_span, num_steps)
        elif method == "heun":
            return _integrate_ode_heun_scan(vector_field, params, z0, x, time_span, num_steps)
        elif method == "rk4":
            return _integrate_ode_rk4_scan(vector_field, params, z0, x, time_span, num_steps)
        elif method == "adaptive":
            # For adaptive method, we need special handling
            return _integrate_ode_adaptive_scan(vector_field, params, z0, x, time_span, num_steps)
        else:
            raise ValueError(f"Unknown integration method: {method}")
    
    elif output_type == "trajectory":
        if method == "euler":
            return _integrate_ode_euler_scan_trajectory(vector_field, params, z0, x, time_span, num_steps)
        elif method == "heun":
            return _integrate_ode_heun_scan_trajectory(vector_field, params, z0, x, time_span, num_steps)
        elif method == "rk4":
            return _integrate_ode_rk4_scan_trajectory(vector_field, params, z0, x, time_span, num_steps)
        elif method == "adaptive":
            return _integrate_ode_adaptive_scan_trajectory(vector_field, params, z0, x, time_span, num_steps)
        else:
            raise ValueError(f"Unknown integration method: {method}")
    
    else:
        raise ValueError(f"Unknown output_type: {output_type}. Must be 'end_point' or 'trajectory'")


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


# =============================================================================
# INDIVIDUAL STEP FUNCTIONS
# =============================================================================

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


# =============================================================================
# SCAN-BASED INTEGRATION FUNCTIONS (END-POINT)
# =============================================================================

def _integrate_ode_euler_scan(
    vector_field: Callable,
    params: Dict[str, Any],
    z0: jnp.ndarray,
    x: jnp.ndarray,
    time_span: Tuple[float, float],
    num_steps: int
) -> jnp.ndarray:
    """JIT-compiled Euler integration using scan."""
    t_start, t_end = time_span
    dt = (t_end - t_start) / num_steps
    
    def euler_step_scan(carry, _):
        z, t = carry
        dz_dt = vector_field(params, z, x, t)
        z_new = z + dt * dz_dt
        t_new = t + dt
        return (z_new, t_new), z_new
    
    # Initial state
    batch_shape = z0.shape[:-1]
    t0 = jnp.full((1,) * len(batch_shape), t_start)
    initial_carry = (z0, t0)
    
    # Scan over integration steps
    (z_final, _), _ = jax.lax.scan(euler_step_scan, initial_carry, None, length=num_steps)
    
    return z_final


def _integrate_ode_heun_scan(
    vector_field: Callable,
    params: Dict[str, Any],
    z0: jnp.ndarray,
    x: jnp.ndarray,
    time_span: Tuple[float, float],
    num_steps: int
) -> jnp.ndarray:
    """JIT-compiled Heun integration using scan."""
    t_start, t_end = time_span
    dt = (t_end - t_start) / num_steps
    
    def heun_step_scan(carry, _):
        z, t = carry
        # First stage
        k1 = vector_field(params, z, x, t)
        
        # Second stage
        z_pred = z + dt * k1
        t_next = t + dt
        k2 = vector_field(params, z_pred, x, t_next)
        
        # Combine stages
        z_new = z + dt * 0.5 * (k1 + k2)
        t_new = t + dt
        
        return (z_new, t_new), z_new
    
    # Initial state
    batch_shape = z0.shape[:-1]
    t0 = jnp.full((1,) * len(batch_shape), t_start)
    initial_carry = (z0, t0)
    
    # Scan over integration steps
    (z_final, _), _ = jax.lax.scan(heun_step_scan, initial_carry, None, length=num_steps)
    
    return z_final


def _integrate_ode_rk4_scan(
    vector_field: Callable,
    params: Dict[str, Any],
    z0: jnp.ndarray,
    x: jnp.ndarray,
    time_span: Tuple[float, float],
    num_steps: int
) -> jnp.ndarray:
    """JIT-compiled RK4 integration using scan."""
    t_start, t_end = time_span
    dt = (t_end - t_start) / num_steps
    
    def rk4_step_scan(carry, _):
        z, t = carry
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
        z_new = z + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
        t_new = t + dt
        
        return (z_new, t_new), z_new
    
    # Initial state
    batch_shape = z0.shape[:-1]
    t0 = jnp.full((1,) * len(batch_shape), t_start)
    initial_carry = (z0, t0)
    
    # Scan over integration steps
    (z_final, _), _ = jax.lax.scan(rk4_step_scan, initial_carry, None, length=num_steps)
    
    return z_final


def _integrate_ode_adaptive_scan(
    vector_field: Callable,
    params: Dict[str, Any],
    z0: jnp.ndarray,
    x: jnp.ndarray,
    time_span: Tuple[float, float],
    max_steps: int
) -> jnp.ndarray:
    """JIT-compiled adaptive integration using scan."""
    t_start, t_end = time_span
    dt = (t_end - t_start) / max_steps
    
    def adaptive_step_scan(carry, _):
        z, t, current_dt = carry
        
        # Check if we've reached the end
        remaining_time = t_end - t
        step_dt = jnp.minimum(current_dt, remaining_time)
        
        # Use adaptive step
        z_new, new_dt = adaptive_step(vector_field, params, z, x, t, step_dt)
        t_new = t + step_dt
        
        return (z_new, t_new, new_dt), z_new
    
    # Initial state
    batch_shape = z0.shape[:-1]
    t0 = jnp.full((1,) * len(batch_shape), t_start)
    initial_carry = (z0, t0, dt)
    
    # Scan over integration steps
    (z_final, _, _), _ = jax.lax.scan(adaptive_step_scan, initial_carry, None, length=max_steps)
    
    return z_final


# =============================================================================
# SCAN-BASED INTEGRATION FUNCTIONS (TRAJECTORY)
# =============================================================================

def _integrate_ode_euler_scan_trajectory(
    vector_field: Callable,
    params: Dict[str, Any],
    z0: jnp.ndarray,
    x: jnp.ndarray,
    time_span: Tuple[float, float],
    num_steps: int
) -> jnp.ndarray:
    """JIT-compiled Euler integration using scan, returning full trajectory."""
    t_start, t_end = time_span
    dt = (t_end - t_start) / num_steps
    
    def euler_step_scan(carry, _):
        z, t = carry
        dz_dt = vector_field(params, z, x, t)
        z_new = z + dt * dz_dt
        t_new = t + dt
        return (z_new, t_new), z_new
    
    # Initial state
    batch_shape = z0.shape[:-1]
    t0 = jnp.full((1,) * len(batch_shape), t_start)
    initial_carry = (z0, t0)
    
    # Scan over integration steps - return full trajectory
    _, trajectory = jax.lax.scan(euler_step_scan, initial_carry, None, length=num_steps)
    
    # Keep trajectory in (num_steps, batch_shape, output_dim) format as documented
    # No transpose needed - scan already returns the correct format
    
    # Prepend the initial state to get the complete trajectory
    # z0 has shape (batch_shape, state_dim), we need (1, batch_shape, state_dim)
    return jnp.concatenate([z0[None, ...], trajectory], axis=0)


def _integrate_ode_heun_scan_trajectory(
    vector_field: Callable,
    params: Dict[str, Any],
    z0: jnp.ndarray,
    x: jnp.ndarray,
    time_span: Tuple[float, float],
    num_steps: int
) -> jnp.ndarray:
    """JIT-compiled Heun integration using scan, returning full trajectory."""
    t_start, t_end = time_span
    dt = (t_end - t_start) / num_steps
    
    def heun_step_scan(carry, _):
        z, t = carry
        # First stage (Euler step)
        k1 = vector_field(params, z, x, t)
        z_euler = z + dt * k1
        t_new = t + dt
        
        # Second stage (Heun correction)
        k2 = vector_field(params, z_euler, x, t_new)
        z_new = z + dt * (k1 + k2) / 2.0
        
        return (z_new, t_new), z_new
    
    # Initial state
    batch_shape = z0.shape[:-1]
    t0 = jnp.full((1,) * len(batch_shape), t_start)
    initial_carry = (z0, t0)
    
    # Scan over integration steps - return full trajectory
    _, trajectory = jax.lax.scan(heun_step_scan, initial_carry, None, length=num_steps)
    
    # Keep trajectory in (num_steps, batch_shape, output_dim) format as documented
    # No transpose needed - scan already returns the correct format
    
    # Prepend the initial state to get the complete trajectory
    # z0 has shape (batch_shape, state_dim), we need (1, batch_shape, state_dim)
    return jnp.concatenate([z0[None, ...], trajectory], axis=0)


def _integrate_ode_rk4_scan_trajectory(
    vector_field: Callable,
    params: Dict[str, Any],
    z0: jnp.ndarray,
    x: jnp.ndarray,
    time_span: Tuple[float, float],
    num_steps: int
) -> jnp.ndarray:
    """JIT-compiled RK4 integration using scan, returning full trajectory."""
    t_start, t_end = time_span
    dt = (t_end - t_start) / num_steps
    
    def rk4_step_scan(carry, _):
        z, t = carry
        
        # RK4 stages
        k1 = vector_field(params, z, x, t)
        k2 = vector_field(params, z + dt * k1 / 2.0, x, t + dt / 2.0)
        k3 = vector_field(params, z + dt * k2 / 2.0, x, t + dt / 2.0)
        k4 = vector_field(params, z + dt * k3, x, t + dt)
        
        # Combine stages
        z_new = z + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        t_new = t + dt
        
        return (z_new, t_new), z_new
    
    # Initial state
    batch_shape = z0.shape[:-1]
    t0 = jnp.full((1,) * len(batch_shape), t_start)
    initial_carry = (z0, t0)
    
    # Scan over integration steps - return full trajectory
    _, trajectory = jax.lax.scan(rk4_step_scan, initial_carry, None, length=num_steps)
    
    # Keep trajectory in (num_steps, batch_shape, output_dim) format as documented
    # No transpose needed - scan already returns the correct format
    
    # Prepend the initial state to get the complete trajectory
    # z0 has shape (batch_shape, state_dim), we need (1, batch_shape, state_dim)
    return jnp.concatenate([z0[None, ...], trajectory], axis=0)


def _integrate_ode_adaptive_scan_trajectory(
    vector_field: Callable,
    params: Dict[str, Any],
    z0: jnp.ndarray,
    x: jnp.ndarray,
    time_span: Tuple[float, float],
    max_steps: int
) -> jnp.ndarray:
    """JIT-compiled adaptive integration using scan, returning full trajectory."""
    t_start, t_end = time_span
    dt = (t_end - t_start) / max_steps
    
    def adaptive_step_scan(carry, _):
        z, t, current_dt = carry
        
        # Check if we've reached the end
        remaining_time = t_end - t
        step_dt = jnp.minimum(current_dt, remaining_time)
        
        # Use adaptive step
        z_new, new_dt = adaptive_step(vector_field, params, z, x, t, step_dt)
        t_new = t + step_dt
        
        return (z_new, t_new, new_dt), z_new
    
    # Initial state
    batch_shape = z0.shape[:-1]
    t0 = jnp.full((1,) * len(batch_shape), t_start)
    initial_carry = (z0, t0, dt)
    
    # Scan over integration steps - return full trajectory
    _, trajectory = jax.lax.scan(adaptive_step_scan, initial_carry, None, length=max_steps)
    
    # Keep trajectory in (max_steps, batch_shape, output_dim) format as documented
    # No transpose needed - scan already returns the correct format
    
    # Prepend the initial state to get the complete trajectory
    # z0 has shape (batch_shape, state_dim), we need (1, batch_shape, state_dim)
    return jnp.concatenate([z0[None, ...], trajectory], axis=0)