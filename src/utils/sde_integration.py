"""
SDE integration methods for NoProp continuous-time variants.

This module provides various numerical integration methods for solving
stochastic differential equations (SDEs) used in NoProp models.
The SDE has the form: dz = f(z, x, t) dt + g(z, x, t) dW
where f is the drift (vector field) and g is the diffusion coefficient.
"""

from typing import Any, Callable, Dict, Tuple, Optional
import jax
import jax.numpy as jnp
import jax.random as jr


# =============================================================================
# MAIN INTEGRATION FUNCTION AND DEFAULTS
# =============================================================================

def integrate_sde(
    drift: Callable,
    diffusion: Callable,
    params: Dict[str, Any],
    z0: jnp.ndarray,
    x: jnp.ndarray,
    time_span: Tuple[float, float],
    num_steps: int,
    prng_key: jr.PRNGKey,
    method: str = "euler",
    output_type: str = "end_point"
) -> jnp.ndarray:
    """Integrate an SDE using the specified method.
    
    This function integrates the SDE dz = f(z, x, t) dt + g(z, x, t) dW
    from t_start to t_end using the specified numerical method with scan-based
    implementation for better JIT compilation.
    
    Args:
        drift: Function that computes the drift term f(z, x, t)
        diffusion: Function that computes the diffusion coefficient g(z, x, t)
        params: Model parameters
        z0: Initial state [batch_size, state_dim]
        x: Input data [batch_size, ...]
        time_span: Tuple of (start_time, end_time)
        num_steps: Number of integration steps
        prng_key: JAX PRNG key for generating random noise
        method: Integration method ("euler", "heun", "rk4", "midpoint", "adaptive")
        output_type: Type of output ("end_point" or "trajectory")
        
    Returns:
        If output_type="end_point": Final state [batch_size, state_dim]
        If output_type="trajectory": Full trajectory [num_steps+1, batch_size, state_dim]
    """
    # Use scan-based JIT-compiled integration functions for better performance
    if output_type == "end_point":
        if method == "euler":
            return _integrate_sde_euler_scan(drift, diffusion, params, z0, x, time_span, num_steps, prng_key)
        elif method == "heun":
            return _integrate_sde_heun_scan(drift, diffusion, params, z0, x, time_span, num_steps, prng_key)
        elif method == "rk4":
            return _integrate_sde_rk4_scan(drift, diffusion, params, z0, x, time_span, num_steps, prng_key)
        elif method == "adaptive":
            return _integrate_sde_adaptive_scan(drift, diffusion, params, z0, x, time_span, num_steps, prng_key)
        elif method == "midpoint":
            return _integrate_sde_midpoint_scan(drift, diffusion, params, z0, x, time_span, num_steps, prng_key)
        else:
            raise ValueError(f"Unknown integration method: {method}")
    
    elif output_type == "trajectory":
        if method == "euler":
            return _integrate_sde_euler_scan_trajectory(drift, diffusion, params, z0, x, time_span, num_steps, prng_key)
        elif method == "heun":
            return _integrate_sde_heun_scan_trajectory(drift, diffusion, params, z0, x, time_span, num_steps, prng_key)
        elif method == "rk4":
            return _integrate_sde_rk4_scan_trajectory(drift, diffusion, params, z0, x, time_span, num_steps, prng_key)
        elif method == "adaptive":
            return _integrate_sde_adaptive_scan_trajectory(drift, diffusion, params, z0, x, time_span, num_steps, prng_key)
        elif method == "midpoint":
            return _integrate_sde_midpoint_scan_trajectory(drift, diffusion, params, z0, x, time_span, num_steps, prng_key)
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

def euler_maruyama_step(
    drift: Callable,
    diffusion: Callable,
    params: Dict[str, Any],
    z: jnp.ndarray,
    x: jnp.ndarray,
    t: jnp.ndarray,
    dt: float,
    prng_key: jr.PRNGKey
) -> jnp.ndarray:
    """Single Euler-Maruyama integration step.
    
    This implements the Euler-Maruyama method:
    z_{t+dt} = z_t + f(z_t, x, t) * dt + g(z_t, x, t) * sqrt(dt) * N(0,1)
    
    Args:
        drift: Function that computes drift f(z, x, t)
        diffusion: Function that computes diffusion coefficient g(z, x, t)
        params: Model parameters
        z: Current state [batch_size, state_dim]
        x: Input data [batch_size, ...]
        t: Current time [batch_size]
        dt: Time step size
        prng_key: JAX PRNG key for generating noise
        
    Returns:
        Updated state [batch_size, state_dim]
    """
    # Compute drift and diffusion
    drift_term = drift(params, z, x, t)
    diffusion_coeff = diffusion(params, z, x, t)
    
    # Generate Wiener process increment: dW ~ N(0, dt)
    # Shape should match z
    batch_shape = z.shape[:-1]
    state_dim = z.shape[-1]
    noise = jr.normal(prng_key, shape=batch_shape + (state_dim,))
    
    # Euler-Maruyama update
    sqrt_dt = jnp.sqrt(dt)
    return z + drift_term * dt + diffusion_coeff * sqrt_dt * noise


def heun_step_sde(
    drift: Callable,
    diffusion: Callable,
    params: Dict[str, Any],
    z: jnp.ndarray,
    x: jnp.ndarray,
    t: jnp.ndarray,
    dt: float,
    prng_key: jr.PRNGKey
) -> jnp.ndarray:
    """Single Heun integration step for SDEs (2nd order Runge-Kutta).
    
    This implements the Heun method for SDEs:
    1. k1_drift = f(z_t, x, t), k1_diff = g(z_t, x, t)
    2. z_pred = z_t + k1_drift * dt + k1_diff * sqrt(dt) * N(0,1)
    3. k2_drift = f(z_pred, x, t + dt), k2_diff = g(z_pred, x, t + dt)
    4. z_{t+dt} = z_t + dt/2 * (k1_drift + k2_drift) + sqrt(dt)/2 * (k1_diff + k2_diff) * N(0,1)
    
    Args:
        drift: Function that computes drift f(z, x, t)
        diffusion: Function that computes diffusion coefficient g(z, x, t)
        params: Model parameters
        z: Current state [batch_size, state_dim]
        x: Input data [batch_size, ...]
        t: Current time [batch_size]
        dt: Time step size
        prng_key: JAX PRNG key for generating noise
        
    Returns:
        Updated state [batch_size, state_dim]
    """
    # Generate noise (same for both stages)
    batch_shape = z.shape[:-1]
    state_dim = z.shape[-1]
    noise = jr.normal(prng_key, shape=batch_shape + (state_dim,))
    sqrt_dt = jnp.sqrt(dt)
    
    # First stage
    k1_drift = drift(params, z, x, t)
    k1_diff = diffusion(params, z, x, t)
    
    # Predictor step
    z_pred = z + k1_drift * dt + k1_diff * sqrt_dt * noise
    t_next = t + dt
    
    # Second stage
    k2_drift = drift(params, z_pred, x, t_next)
    k2_diff = diffusion(params, z_pred, x, t_next)
    
    # Combine stages (average drift, average diffusion)
    return z + dt * 0.5 * (k1_drift + k2_drift) + sqrt_dt * 0.5 * (k1_diff + k2_diff) * noise


def rk4_step_sde(
    drift: Callable,
    diffusion: Callable,
    params: Dict[str, Any],
    z: jnp.ndarray,
    x: jnp.ndarray,
    t: jnp.ndarray,
    dt: float,
    prng_key: jr.PRNGKey
) -> jnp.ndarray:
    """Single 4th order Runge-Kutta integration step for SDEs.
    
    This implements an RK4-like method for SDEs with noise:
    1. k1_drift = f(z_t, x, t), k1_diff = g(z_t, x, t)
    2. k2_drift = f(z_t + dt/2*k1_drift, x, t + dt/2), k2_diff = g(...)
    3. k3_drift = f(z_t + dt/2*k2_drift, x, t + dt/2), k3_diff = g(...)
    4. k4_drift = f(z_t + dt*k3_drift, x, t + dt), k4_diff = g(...)
    5. z_{t+dt} = z_t + dt/6 * (k1_drift + 2*k2_drift + 2*k3_drift + k4_drift)
                  + sqrt(dt)/6 * (k1_diff + 2*k2_diff + 2*k3_diff + k4_diff) * N(0,1)
    
    Args:
        drift: Function that computes drift f(z, x, t)
        diffusion: Function that computes diffusion coefficient g(z, x, t)
        params: Model parameters
        z: Current state [batch_size, state_dim]
        x: Input data [batch_size, ...]
        t: Current time [batch_size]
        dt: Time step size
        prng_key: JAX PRNG key for generating noise
        
    Returns:
        Updated state [batch_size, state_dim]
    """
    # Generate noise (same for all stages)
    batch_shape = z.shape[:-1]
    state_dim = z.shape[-1]
    noise = jr.normal(prng_key, shape=batch_shape + (state_dim,))
    sqrt_dt = jnp.sqrt(dt)
    
    # Stage 1
    k1_drift = drift(params, z, x, t)
    k1_diff = diffusion(params, z, x, t)
    
    # Stage 2
    z2 = z + dt * 0.5 * k1_drift
    t2 = t + dt * 0.5
    k2_drift = drift(params, z2, x, t2)
    k2_diff = diffusion(params, z2, x, t2)
    
    # Stage 3
    z3 = z + dt * 0.5 * k2_drift
    k3_drift = drift(params, z3, x, t2)
    k3_diff = diffusion(params, z3, x, t2)
    
    # Stage 4
    z4 = z + dt * k3_drift
    t4 = t + dt
    k4_drift = drift(params, z4, x, t4)
    k4_diff = diffusion(params, z4, x, t4)
    
    # Combine stages
    drift_update = dt * (k1_drift + 2*k2_drift + 2*k3_drift + k4_drift) / 6.0
    diffusion_update = sqrt_dt * (k1_diff + 2*k2_diff + 2*k3_diff + k4_diff) / 6.0 * noise
    
    return z + drift_update + diffusion_update


def midpoint_step_sde(
    drift: Callable,
    diffusion: Callable,
    params: Dict[str, Any],
    z: jnp.ndarray,
    x: jnp.ndarray,
    t: jnp.ndarray,
    dt: float,
    prng_key: jr.PRNGKey
) -> jnp.ndarray:
    """Single midpoint integration step for SDEs.
    
    This evaluates the drift and diffusion at the midpoint time:
    z_{t+dt} = z_t + f(z_t, x, t + dt/2) * dt + g(z_t, x, t + dt/2) * sqrt(dt) * N(0,1)
    
    Args:
        drift: Function that computes drift f(z, x, t)
        diffusion: Function that computes diffusion coefficient g(z, x, t)
        params: Model parameters
        z: Current state [batch_size, state_dim]
        x: Input data [batch_size, ...]
        t: Current time [batch_size]
        dt: Time step size
        prng_key: JAX PRNG key for generating noise
        
    Returns:
        Updated state [batch_size, state_dim]
    """
    # Compute drift and diffusion at midpoint time
    drift_term = drift(params, z, x, t + 0.5*dt)
    diffusion_coeff = diffusion(params, z, x, t + 0.5*dt)
    
    # Generate Wiener process increment
    batch_shape = z.shape[:-1]
    state_dim = z.shape[-1]
    noise = jr.normal(prng_key, shape=batch_shape + (state_dim,))
    
    # Midpoint update (like Euler but evaluated at midpoint time)
    sqrt_dt = jnp.sqrt(dt)
    return z + drift_term * dt + diffusion_coeff * sqrt_dt * noise


def adaptive_step_sde(
    drift: Callable,
    diffusion: Callable,
    params: Dict[str, Any],
    z: jnp.ndarray,
    x: jnp.ndarray,
    t: jnp.ndarray,
    dt: float,
    prng_key: jr.PRNGKey,
    tolerance: float = 1e-6
) -> Tuple[jnp.ndarray, float]:
    """Adaptive step size integration for SDEs.
    
    This uses error estimation to adaptively choose step sizes.
    It compares a full step with two half steps to estimate error.
    
    Args:
        drift: Function that computes drift f(z, x, t)
        diffusion: Function that computes diffusion coefficient g(z, x, t)
        params: Model parameters
        z: Current state [batch_size, state_dim]
        x: Input data [batch_size, ...]
        t: Current time [batch_size]
        dt: Initial time step size
        prng_key: JAX PRNG key for generating noise
        tolerance: Error tolerance for adaptive stepping
        
    Returns:
        Tuple of (updated_state, next_step_size)
    """
    # Split key for full step and half steps
    key1, key2, key3 = jr.split(prng_key, 3)
    
    # Full step
    z_full = heun_step_sde(drift, diffusion, params, z, x, t, dt, key1)
    
    # Two half steps
    z_half1 = heun_step_sde(drift, diffusion, params, z, x, t, dt/2, key2)
    z_half2 = heun_step_sde(drift, diffusion, params, z_half1, x, t + dt/2, dt/2, key3)
    
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

def _integrate_sde_euler_scan(
    drift: Callable,
    diffusion: Callable,
    params: Dict[str, Any],
    z0: jnp.ndarray,
    x: jnp.ndarray,
    time_span: Tuple[float, float],
    num_steps: int,
    prng_key: jr.PRNGKey
) -> jnp.ndarray:
    """JIT-compiled Euler-Maruyama integration using scan."""
    t_start, t_end = time_span
    dt = (t_end - t_start) / num_steps
    
    # Split keys for each step
    keys = jr.split(prng_key, num_steps)
    
    def euler_step_scan(carry, key):
        z, t = carry
        drift_term = drift(params, z, x, t)
        diffusion_coeff = diffusion(params, z, x, t)
        
        # Generate noise
        batch_shape = z.shape[:-1]
        state_dim = z.shape[-1]
        noise = jr.normal(key, shape=batch_shape + (state_dim,))
        
        # Euler-Maruyama update
        sqrt_dt = jnp.sqrt(dt)
        z_new = z + drift_term * dt + diffusion_coeff * sqrt_dt * noise
        t_new = t + dt
        
        return (z_new, t_new), z_new
    
    # Initial state
    batch_shape = z0.shape[:-1]
    t0 = jnp.full((1,) * len(batch_shape), t_start)
    initial_carry = (z0, t0)
    
    # Scan over integration steps
    (z_final, _), _ = jax.lax.scan(euler_step_scan, initial_carry, keys, length=num_steps)
    
    return z_final


def _integrate_sde_heun_scan(
    drift: Callable,
    diffusion: Callable,
    params: Dict[str, Any],
    z0: jnp.ndarray,
    x: jnp.ndarray,
    time_span: Tuple[float, float],
    num_steps: int,
    prng_key: jr.PRNGKey
) -> jnp.ndarray:
    """JIT-compiled Heun integration for SDEs using scan."""
    t_start, t_end = time_span
    dt = (t_end - t_start) / num_steps
    
    # Split keys for each step
    keys = jr.split(prng_key, num_steps)
    
    def heun_step_scan(carry, key):
        z, t = carry
        sqrt_dt = jnp.sqrt(dt)
        
        # Generate noise
        batch_shape = z.shape[:-1]
        state_dim = z.shape[-1]
        noise = jr.normal(key, shape=batch_shape + (state_dim,))
        
        # First stage
        k1_drift = drift(params, z, x, t)
        k1_diff = diffusion(params, z, x, t)
        
        # Predictor step
        z_pred = z + k1_drift * dt + k1_diff * sqrt_dt * noise
        t_next = t + dt
        
        # Second stage
        k2_drift = drift(params, z_pred, x, t_next)
        k2_diff = diffusion(params, z_pred, x, t_next)
        
        # Combine stages
        z_new = z + dt * 0.5 * (k1_drift + k2_drift) + sqrt_dt * 0.5 * (k1_diff + k2_diff) * noise
        t_new = t + dt
        
        return (z_new, t_new), z_new
    
    # Initial state
    batch_shape = z0.shape[:-1]
    t0 = jnp.full((1,) * len(batch_shape), t_start)
    initial_carry = (z0, t0)
    
    # Scan over integration steps
    (z_final, _), _ = jax.lax.scan(heun_step_scan, initial_carry, keys, length=num_steps)
    
    return z_final


def _integrate_sde_rk4_scan(
    drift: Callable,
    diffusion: Callable,
    params: Dict[str, Any],
    z0: jnp.ndarray,
    x: jnp.ndarray,
    time_span: Tuple[float, float],
    num_steps: int,
    prng_key: jr.PRNGKey
) -> jnp.ndarray:
    """JIT-compiled RK4 integration for SDEs using scan."""
    t_start, t_end = time_span
    dt = (t_end - t_start) / num_steps
    
    # Split keys for each step
    keys = jr.split(prng_key, num_steps)
    
    def rk4_step_scan(carry, key):
        z, t = carry
        sqrt_dt = jnp.sqrt(dt)
        
        # Generate noise
        batch_shape = z.shape[:-1]
        state_dim = z.shape[-1]
        noise = jr.normal(key, shape=batch_shape + (state_dim,))
        
        # Stage 1
        k1_drift = drift(params, z, x, t)
        k1_diff = diffusion(params, z, x, t)
        
        # Stage 2
        z2 = z + dt * 0.5 * k1_drift
        t2 = t + dt * 0.5
        k2_drift = drift(params, z2, x, t2)
        k2_diff = diffusion(params, z2, x, t2)
        
        # Stage 3
        z3 = z + dt * 0.5 * k2_drift
        k3_drift = drift(params, z3, x, t2)
        k3_diff = diffusion(params, z3, x, t2)
        
        # Stage 4
        z4 = z + dt * k3_drift
        t4 = t + dt
        k4_drift = drift(params, z4, x, t4)
        k4_diff = diffusion(params, z4, x, t4)
        
        # Combine stages
        drift_update = dt * (k1_drift + 2*k2_drift + 2*k3_drift + k4_drift) / 6.0
        diffusion_update = sqrt_dt * (k1_diff + 2*k2_diff + 2*k3_diff + k4_diff) / 6.0 * noise
        
        z_new = z + drift_update + diffusion_update
        t_new = t + dt
        
        return (z_new, t_new), z_new
    
    # Initial state
    batch_shape = z0.shape[:-1]
    t0 = jnp.full((1,) * len(batch_shape), t_start)
    initial_carry = (z0, t0)
    
    # Scan over integration steps
    (z_final, _), _ = jax.lax.scan(rk4_step_scan, initial_carry, keys, length=num_steps)
    
    return z_final


def _integrate_sde_midpoint_scan(
    drift: Callable,
    diffusion: Callable,
    params: Dict[str, Any],
    z0: jnp.ndarray,
    x: jnp.ndarray,
    time_span: Tuple[float, float],
    num_steps: int,
    prng_key: jr.PRNGKey
) -> jnp.ndarray:
    """JIT-compiled midpoint integration for SDEs using scan."""
    t_start, t_end = time_span
    dt = (t_end - t_start) / num_steps
    
    # Split keys for each step
    keys = jr.split(prng_key, num_steps)
    
    def midpoint_step_scan(carry, key):
        z, t = carry
        # Evaluate drift and diffusion at midpoint time
        drift_term = drift(params, z, x, t + 0.5*dt)
        diffusion_coeff = diffusion(params, z, x, t + 0.5*dt)
        
        # Generate noise
        batch_shape = z.shape[:-1]
        state_dim = z.shape[-1]
        noise = jr.normal(key, shape=batch_shape + (state_dim,))
        
        # Midpoint update (like Euler but evaluated at midpoint time)
        sqrt_dt = jnp.sqrt(dt)
        z_new = z + drift_term * dt + diffusion_coeff * sqrt_dt * noise
        t_new = t + dt
        
        return (z_new, t_new), z_new
    
    # Initial state
    batch_shape = z0.shape[:-1]
    t0 = jnp.full((1,) * len(batch_shape), t_start)
    initial_carry = (z0, t0)
    
    # Scan over integration steps
    (z_final, _), _ = jax.lax.scan(midpoint_step_scan, initial_carry, keys, length=num_steps)
    
    return z_final


def _integrate_sde_adaptive_scan(
    drift: Callable,
    diffusion: Callable,
    params: Dict[str, Any],
    z0: jnp.ndarray,
    x: jnp.ndarray,
    time_span: Tuple[float, float],
    max_steps: int,
    prng_key: jr.PRNGKey
) -> jnp.ndarray:
    """JIT-compiled adaptive integration for SDEs using scan."""
    t_start, t_end = time_span
    dt = (t_end - t_start) / max_steps
    
    # Split keys for each step (we'll need multiple keys per step for adaptive)
    keys = jr.split(prng_key, max_steps * 3)  # 3 keys per step (full, half1, half2)
    
    def adaptive_step_scan(carry, idx):
        z, t, current_dt = carry
        
        # Check if we've reached the end
        remaining_time = t_end - t
        step_dt = jnp.minimum(current_dt, remaining_time)
        
        # Get keys for this step
        key_idx = idx * 3
        key1, key2, key3 = keys[key_idx], keys[key_idx + 1], keys[key_idx + 2]
        
        # Use adaptive step
        z_new, new_dt = adaptive_step_sde(drift, diffusion, params, z, x, t, step_dt, key1, tolerance=1e-6)
        t_new = t + step_dt
        
        return (z_new, t_new, new_dt), z_new
    
    # Initial state
    batch_shape = z0.shape[:-1]
    t0 = jnp.full((1,) * len(batch_shape), t_start)
    initial_carry = (z0, t0, dt)
    
    # Scan over integration steps
    indices = jnp.arange(max_steps)
    (z_final, _, _), _ = jax.lax.scan(adaptive_step_scan, initial_carry, indices, length=max_steps)
    
    return z_final


# =============================================================================
# SCAN-BASED INTEGRATION FUNCTIONS (TRAJECTORY)
# =============================================================================

def _integrate_sde_euler_scan_trajectory(
    drift: Callable,
    diffusion: Callable,
    params: Dict[str, Any],
    z0: jnp.ndarray,
    x: jnp.ndarray,
    time_span: Tuple[float, float],
    num_steps: int,
    prng_key: jr.PRNGKey
) -> jnp.ndarray:
    """JIT-compiled Euler-Maruyama integration using scan, returning full trajectory."""
    t_start, t_end = time_span
    dt = (t_end - t_start) / num_steps
    
    # Split keys for each step
    keys = jr.split(prng_key, num_steps)
    
    def euler_step_scan(carry, key):
        z, t = carry
        drift_term = drift(params, z, x, t)
        diffusion_coeff = diffusion(params, z, x, t)
        
        # Generate noise
        batch_shape = z.shape[:-1]
        state_dim = z.shape[-1]
        noise = jr.normal(key, shape=batch_shape + (state_dim,))
        
        # Euler-Maruyama update
        sqrt_dt = jnp.sqrt(dt)
        z_new = z + drift_term * dt + diffusion_coeff * sqrt_dt * noise
        t_new = t + dt
        
        return (z_new, t_new), z_new
    
    # Initial state
    batch_shape = z0.shape[:-1]
    t0 = jnp.full((1,) * len(batch_shape), t_start)
    initial_carry = (z0, t0)
    
    # Scan over integration steps - return full trajectory
    _, trajectory = jax.lax.scan(euler_step_scan, initial_carry, keys, length=num_steps)
    
    # Prepend the initial state to get the complete trajectory
    return jnp.concatenate([z0[None, ...], trajectory], axis=0)


def _integrate_sde_heun_scan_trajectory(
    drift: Callable,
    diffusion: Callable,
    params: Dict[str, Any],
    z0: jnp.ndarray,
    x: jnp.ndarray,
    time_span: Tuple[float, float],
    num_steps: int,
    prng_key: jr.PRNGKey
) -> jnp.ndarray:
    """JIT-compiled Heun integration for SDEs using scan, returning full trajectory."""
    t_start, t_end = time_span
    dt = (t_end - t_start) / num_steps
    
    # Split keys for each step
    keys = jr.split(prng_key, num_steps)
    
    def heun_step_scan(carry, key):
        z, t = carry
        sqrt_dt = jnp.sqrt(dt)
        
        # Generate noise
        batch_shape = z.shape[:-1]
        state_dim = z.shape[-1]
        noise = jr.normal(key, shape=batch_shape + (state_dim,))
        
        # First stage
        k1_drift = drift(params, z, x, t)
        k1_diff = diffusion(params, z, x, t)
        
        # Predictor step
        z_pred = z + k1_drift * dt + k1_diff * sqrt_dt * noise
        t_new = t + dt
        
        # Second stage
        k2_drift = drift(params, z_pred, x, t_new)
        k2_diff = diffusion(params, z_pred, x, t_new)
        
        # Combine stages
        z_new = z + dt * 0.5 * (k1_drift + k2_drift) + sqrt_dt * 0.5 * (k1_diff + k2_diff) * noise
        
        return (z_new, t_new), z_new
    
    # Initial state
    batch_shape = z0.shape[:-1]
    t0 = jnp.full((1,) * len(batch_shape), t_start)
    initial_carry = (z0, t0)
    
    # Scan over integration steps - return full trajectory
    _, trajectory = jax.lax.scan(heun_step_scan, initial_carry, keys, length=num_steps)
    
    # Prepend the initial state to get the complete trajectory
    return jnp.concatenate([z0[None, ...], trajectory], axis=0)


def _integrate_sde_rk4_scan_trajectory(
    drift: Callable,
    diffusion: Callable,
    params: Dict[str, Any],
    z0: jnp.ndarray,
    x: jnp.ndarray,
    time_span: Tuple[float, float],
    num_steps: int,
    prng_key: jr.PRNGKey
) -> jnp.ndarray:
    """JIT-compiled RK4 integration for SDEs using scan, returning full trajectory."""
    t_start, t_end = time_span
    dt = (t_end - t_start) / num_steps
    
    # Split keys for each step
    keys = jr.split(prng_key, num_steps)
    
    def rk4_step_scan(carry, key):
        z, t = carry
        sqrt_dt = jnp.sqrt(dt)
        
        # Generate noise
        batch_shape = z.shape[:-1]
        state_dim = z.shape[-1]
        noise = jr.normal(key, shape=batch_shape + (state_dim,))
        
        # Stage 1
        k1_drift = drift(params, z, x, t)
        k1_diff = diffusion(params, z, x, t)
        
        # Stage 2
        z2 = z + dt * 0.5 * k1_drift
        t2 = t + dt * 0.5
        k2_drift = drift(params, z2, x, t2)
        k2_diff = diffusion(params, z2, x, t2)
        
        # Stage 3
        z3 = z + dt * 0.5 * k2_drift
        k3_drift = drift(params, z3, x, t2)
        k3_diff = diffusion(params, z3, x, t2)
        
        # Stage 4
        z4 = z + dt * k3_drift
        t4 = t + dt
        k4_drift = drift(params, z4, x, t4)
        k4_diff = diffusion(params, z4, x, t4)
        
        # Combine stages
        drift_update = dt * (k1_drift + 2*k2_drift + 2*k3_drift + k4_drift) / 6.0
        diffusion_update = sqrt_dt * (k1_diff + 2*k2_diff + 2*k3_diff + k4_diff) / 6.0 * noise
        
        z_new = z + drift_update + diffusion_update
        t_new = t + dt
        
        return (z_new, t_new), z_new
    
    # Initial state
    batch_shape = z0.shape[:-1]
    t0 = jnp.full((1,) * len(batch_shape), t_start)
    initial_carry = (z0, t0)
    
    # Scan over integration steps - return full trajectory
    _, trajectory = jax.lax.scan(rk4_step_scan, initial_carry, keys, length=num_steps)
    
    # Prepend the initial state to get the complete trajectory
    return jnp.concatenate([z0[None, ...], trajectory], axis=0)


def _integrate_sde_adaptive_scan_trajectory(
    drift: Callable,
    diffusion: Callable,
    params: Dict[str, Any],
    z0: jnp.ndarray,
    x: jnp.ndarray,
    time_span: Tuple[float, float],
    max_steps: int,
    prng_key: jr.PRNGKey
) -> jnp.ndarray:
    """JIT-compiled adaptive integration for SDEs using scan, returning full trajectory."""
    t_start, t_end = time_span
    dt = (t_end - t_start) / max_steps
    
    # Split keys for each step (we'll need multiple keys per step for adaptive)
    keys = jr.split(prng_key, max_steps * 3)  # 3 keys per step
    
    def adaptive_step_scan(carry, idx):
        z, t, current_dt = carry
        
        # Check if we've reached the end
        remaining_time = t_end - t
        step_dt = jnp.minimum(current_dt, remaining_time)
        
        # Get keys for this step
        key_idx = idx * 3
        key1, key2, key3 = keys[key_idx], keys[key_idx + 1], keys[key_idx + 2]
        
        # Use adaptive step
        z_new, new_dt = adaptive_step_sde(drift, diffusion, params, z, x, t, step_dt, key1, tolerance=1e-6)
        t_new = t + step_dt
        
        return (z_new, t_new, new_dt), z_new
    
    # Initial state
    batch_shape = z0.shape[:-1]
    t0 = jnp.full((1,) * len(batch_shape), t_start)
    initial_carry = (z0, t0, dt)
    
    # Scan over integration steps - return full trajectory
    indices = jnp.arange(max_steps)
    _, trajectory = jax.lax.scan(adaptive_step_scan, initial_carry, indices, length=max_steps)
    
    # Prepend the initial state to get the complete trajectory
    return jnp.concatenate([z0[None, ...], trajectory], axis=0)


def _integrate_sde_midpoint_scan_trajectory(
    drift: Callable,
    diffusion: Callable,
    params: Dict[str, Any],
    z0: jnp.ndarray,
    x: jnp.ndarray,
    time_span: Tuple[float, float],
    num_steps: int,
    prng_key: jr.PRNGKey
) -> jnp.ndarray:
    """JIT-compiled midpoint integration for SDEs using scan that returns full trajectory."""
    t_start, t_end = time_span
    dt = (t_end - t_start) / num_steps
    
    # Split keys for each step
    keys = jr.split(prng_key, num_steps)
    
    def midpoint_step_scan(carry, key):
        z, t = carry
        # Evaluate drift and diffusion at midpoint time
        drift_term = drift(params, z, x, t + 0.5*dt)
        diffusion_coeff = diffusion(params, z, x, t + 0.5*dt)
        
        # Generate noise
        batch_shape = z.shape[:-1]
        state_dim = z.shape[-1]
        noise = jr.normal(key, shape=batch_shape + (state_dim,))
        
        # Midpoint update (like Euler but evaluated at midpoint time)
        sqrt_dt = jnp.sqrt(dt)
        z_new = z + drift_term * dt + diffusion_coeff * sqrt_dt * noise
        t_new = t + dt
        
        return (z_new, t_new), z_new
    
    # Initial state
    batch_shape = z0.shape[:-1]
    t0 = jnp.full((1,) * len(batch_shape), t_start)
    initial_carry = (z0, t0)
    
    # Scan over integration steps
    _, trajectory = jax.lax.scan(midpoint_step_scan, initial_carry, keys, length=num_steps)
    
    # Prepend the initial state to get the complete trajectory
    return jnp.concatenate([z0[None, ...], trajectory], axis=0)

