# ODE Integration Module

## Overview

I've successfully created a dedicated `ode_integration.py` module that contains all the numerical integration methods for solving neural ordinary differential equations (ODEs) used in NoProp-CT and NoProp-FM variants.

## New File: `src/jax_noprop/ode_integration.py`

### **Integration Methods Available**

#### 1. **Euler Method** (`euler_step`)
- **Order**: 1st order
- **Formula**: `z_{t+dt} = z_t + dt * f(z_t, x, t)`
- **Use case**: Fast training, basic integration

#### 2. **Heun Method** (`heun_step`)
- **Order**: 2nd order (improved Euler)
- **Formula**: 
  ```
  k1 = f(z_t, x, t)
  k2 = f(z_t + dt*k1, x, t + dt)
  z_{t+dt} = z_t + dt/2 * (k1 + k2)
  ```
- **Use case**: More accurate evaluation, good balance of speed/accuracy

#### 3. **Runge-Kutta 4th Order** (`rk4_step`)
- **Order**: 4th order
- **Formula**: Classic RK4 with 4 stages
- **Use case**: High precision integration when needed

#### 4. **Adaptive Step Size** (`adaptive_step`)
- **Method**: Error estimation with adaptive step size
- **Use case**: Automatic step size control for optimal accuracy/efficiency

### **Main Integration Functions**

#### **`integrate_ode()`**
```python
def integrate_ode(
    vector_field: Callable,
    params: Dict[str, Any],
    z0: jnp.ndarray,
    x: jnp.ndarray,
    time_span: Tuple[float, float],
    num_steps: int,
    method: str = "euler"
) -> jnp.ndarray:
```

#### **`integrate_flow()`**
```python
def integrate_flow(
    vector_field: Callable,
    params: Dict[str, Any],
    z0: jnp.ndarray,
    x: jnp.ndarray,
    time_span: Tuple[float, float],
    num_steps: int,
    method: str = "euler"
) -> jnp.ndarray:
```
- Alias for `integrate_ode()` for flow matching terminology

### **Default Configurations**

```python
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
```

## Updated Files

### **NoProp-CT (`noprop_ct.py`)**
- **Removed**: `euler_step()`, `heun_step()`, and complex `integrate_ode()` methods
- **Added**: Import from `ode_integration` module
- **Simplified**: `integrate_ode()` now just calls the module function

**Before:**
```python
def integrate_ode(self, params, z0, x, t_span, num_steps):
    # 30+ lines of integration logic
    for _ in range(num_steps):
        if self.integration_method == "euler":
            z = self.euler_step(params, z, x, t, dt)
        elif self.integration_method == "heun":
            z = self.heun_step(params, z, x, t, dt)
        # ... more logic
```

**After:**
```python
def integrate_ode(self, params, z0, x, t_span, num_steps):
    return integrate_ode(
        vector_field=self.vector_field,
        params=params,
        z0=z0,
        x=x,
        time_span=t_span,
        num_steps=num_steps,
        method=self.integration_method
    )
```

### **NoProp-FM (`noprop_fm.py`)**
- **Removed**: `euler_step()`, `heun_step()`, and complex `integrate_flow()` methods
- **Added**: Import from `ode_integration` module
- **Simplified**: `integrate_flow()` now uses the module function

### **Package Exports (`__init__.py`)**
- **Added**: All ODE integration functions to package exports
- **Available**: `euler_step`, `heun_step`, `rk4_step`, `adaptive_step`, `integrate_ode`, `integrate_flow`

## Benefits

### **1. Code Organization**
- **Separation of concerns**: ODE integration logic separated from NoProp algorithms
- **Reusability**: Integration methods can be used by other modules
- **Maintainability**: Centralized location for all ODE integration code

### **2. Enhanced Functionality**
- **More methods**: Added RK4 and adaptive step size methods
- **Better accuracy**: Higher-order methods available for evaluation
- **Flexibility**: Easy to switch between integration methods

### **3. Cleaner Code**
- **Reduced duplication**: No more duplicate ODE code in NoProp-CT and NoProp-FM
- **Simplified classes**: NoProp classes are now much cleaner
- **Better testing**: ODE methods can be tested independently

### **4. Extensibility**
- **Easy to add new methods**: New integration schemes can be added to the module
- **Configurable**: Default configurations for different use cases
- **Modular**: Can be used independently of NoProp

## Usage Examples

### **Direct Usage**
```python
from jax_noprop.ode_integration import integrate_ode, euler_step

# Define vector field
def vector_field(params, z, x, t):
    return -z  # Simple exponential decay

# Integrate ODE
z_final = integrate_ode(
    vector_field=vector_field,
    params={},
    z0=jnp.array([[1.0, 2.0]]),
    x=jnp.ones((1, 10)),
    time_span=(0.0, 1.0),
    num_steps=10,
    method="heun"
)
```

### **With NoProp**
```python
from jax_noprop import NoPropCT

# NoProp-CT automatically uses the ODE integration module
noprop_ct = NoPropCT(model=model, integration_method="heun")
# The integration is handled internally using the module
```

## Testing Results

- **All tests pass** (5/5) ✅
- **ODE integration methods work correctly** ✅
- **NoProp-CT and NoProp-FM still function properly** ✅
- **Shape consistency maintained** ✅

## Conclusion

The new `ode_integration.py` module provides:
- **Clean separation** of ODE integration logic
- **Enhanced functionality** with multiple integration methods
- **Better code organization** and maintainability
- **Reusable components** for other projects
- **Full backward compatibility** with existing NoProp implementations

The NoProp-CT and NoProp-FM classes are now much cleaner and focused on their core algorithms, while the ODE integration is handled by a dedicated, well-tested module.
