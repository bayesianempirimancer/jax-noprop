"""
Gradient and Hessian computation utilities for neural networks.

This module provides precompiled vmapped functions for efficiently computing
gradients and Hessians of neural networks, which is critical for runtime
performance in logZ models where these computations are performed frequently.

Key features:
- Precompiled JIT functions for maximum performance
- Support for both diagonal and full Hessian computation
- Batch processing with vmap for vectorized operations
- Flexible interface for different network architectures
- Memory-efficient implementations
"""

import jax
import jax.numpy as jnp
from typing import Dict, Callable, Optional, Union, Tuple
from flax.core import FrozenDict


class GradientHessianUtils:
    """
    Efficient gradient and Hessian computation for neural networks.
    
    This class provides precompiled functions for computing gradients and Hessians
    of neural networks, optimized for batch processing and runtime performance.
    """
    
    def __init__(self, 
                 model_apply_fn: Callable,
                 hessian_method: str = 'full',
                 compile_functions: bool = True):
        """
        Initialize the gradient/hessian utilities.
        
        Args:
            model_apply_fn: The model's apply function (e.g., model.apply)
            hessian_method: Method for Hessian computation ('diagonal' or 'full')
            compile_functions: Whether to JIT compile the functions for performance
        """
        self.model_apply_fn = model_apply_fn
        self.hessian_method = hessian_method
        self.compile_functions = compile_functions
        
        # Precompiled functions (will be set after first use)
        self._compiled_gradient_fn = None
        self._compiled_hessian_fn = None
        self._last_params_hash = None
        
    def _get_params_hash(self, params: Union[Dict, FrozenDict]) -> int:
        """Get a hash of the parameters for caching compiled functions."""
        # Simple hash based on parameter shapes and a few values
        param_shapes = jax.tree.map(lambda x: x.shape, params)
        param_sample = jax.tree.map(lambda x: x.flatten()[0] if x.size > 0 else 0.0, params)
        return hash((str(param_shapes), str(param_sample)))
    
    def _compile_gradient_function(self, params: Union[Dict, FrozenDict]) -> Callable:
        """
        Compile gradient function for the given parameters.
        
        Args:
            params: Model parameters
            
        Returns:
            Compiled gradient function
        """
        def single_gradient_fn(eta_single: jnp.ndarray) -> jnp.ndarray:
            """Compute gradient for a single eta value."""
            def log_normalizer(eta: jnp.ndarray) -> jnp.ndarray:
                output = self.model_apply_fn(params, eta, training=False)
                # Assert that model returns expected shape (1,) for single eta vector in LogZ models
                assert output.shape == (1,), f"LogZ model must return shape (1,) for single eta vector, got {output.shape}. Use nn.Dense(1) and return output directly (don't squeeze)."
                return jnp.squeeze(output, axis=-1)  # Extract scalar from (1,) shape
            return jax.grad(log_normalizer)(eta_single)
        
        def batch_gradient_fn(eta_batch: jnp.ndarray) -> jnp.ndarray:
            """Compute gradients for a batch of eta values."""
            # Handle arbitrary batch shapes by flattening and reshaping
            original_shape = eta_batch.shape
            eta_flat = eta_batch.reshape(-1, eta_batch.shape[-1])
            gradients_flat = jax.vmap(single_gradient_fn)(eta_flat)
            return gradients_flat.reshape(original_shape)
        
        if self.compile_functions:
            return jax.jit(batch_gradient_fn)
        else:
            return batch_gradient_fn
    
    def _compile_hessian_function(self, params: Union[Dict, FrozenDict]) -> Callable:
        """
        Compile Hessian function for the given parameters.
        
        Args:
            params: Model parameters
            
        Returns:
            Compiled Hessian function
        """
        if self.hessian_method == 'diagonal':
            def single_hessian_diag_fn(eta_single: jnp.ndarray) -> jnp.ndarray:
                """Compute diagonal Hessian for a single eta value."""
                def log_normalizer(eta: jnp.ndarray) -> jnp.ndarray:
                    output = self.model_apply_fn(params, eta, training=False)
                    # Assert that model returns expected shape (1,) for single eta vector in LogZ models
                    assert output.shape == (1,), f"LogZ model must return shape (1,) for single eta vector, got {output.shape}. Use nn.Dense(1) and return output directly (don't squeeze)."
                    return jnp.squeeze(output, axis=-1)  # Extract scalar from (1,) shape
                
                grad_fn = jax.grad(log_normalizer)
                return jnp.diag(jax.jacfwd(grad_fn)(eta_single))
            
            def batch_hessian_diag_fn(eta_batch: jnp.ndarray) -> jnp.ndarray:
                """Compute diagonal Hessians for a batch of eta values."""
                # Handle arbitrary batch shapes by flattening and reshaping
                original_shape = eta_batch.shape
                eta_flat = eta_batch.reshape(-1, eta_batch.shape[-1])
                hessians_flat = jax.vmap(single_hessian_diag_fn)(eta_flat)
                return hessians_flat.reshape(original_shape)
            
            if self.compile_functions:
                return jax.jit(batch_hessian_diag_fn)
            else:
                return batch_hessian_diag_fn
                
        elif self.hessian_method == 'full':
            def single_hessian_full_fn(eta_single: jnp.ndarray) -> jnp.ndarray:
                """Compute full Hessian for a single eta value."""
                def log_normalizer(eta: jnp.ndarray) -> jnp.ndarray:
                    output = self.model_apply_fn(params, eta, training=False)
                    # Assert that model returns expected shape (1,) for single eta vector in LogZ models
                    assert output.shape == (1,), f"LogZ model must return shape (1,) for single eta vector, got {output.shape}. Use nn.Dense(1) and return output directly (don't squeeze)."
                    return jnp.squeeze(output, axis=-1)  # Extract scalar from (1,) shape
                return jax.hessian(log_normalizer)(eta_single)
            
            def batch_hessian_full_fn(eta_batch: jnp.ndarray) -> jnp.ndarray:
                """Compute full Hessians for a batch of eta values."""
                # Handle arbitrary batch shapes by flattening and reshaping
                original_shape = eta_batch.shape
                eta_flat = eta_batch.reshape(-1, eta_batch.shape[-1])
                hessians_flat = jax.vmap(single_hessian_full_fn)(eta_flat)
                # Reshape to original batch shape + (eta_dim, eta_dim)
                return hessians_flat.reshape(original_shape + (eta_batch.shape[-1],))
            
            if self.compile_functions:
                return jax.jit(batch_hessian_full_fn)
            else:
                return batch_hessian_full_fn
        else:
            raise ValueError(f"Unknown hessian_method: {self.hessian_method}")
    
    def compute_gradient(self, 
                        params: Union[Dict, FrozenDict], 
                        eta: jnp.ndarray) -> jnp.ndarray:
        """
        Compute gradient of the log normalizer (expectation of sufficient statistics).
        
        Args:
            params: Model parameters
            eta: Natural parameters [batch_size, eta_dim]
            
        Returns:
            Gradients [batch_size, eta_dim] representing E[T(X)|eta]
        """
        # Check if we need to recompile functions
        params_hash = self._get_params_hash(params)
        if (self._compiled_gradient_fn is None or 
            self._last_params_hash != params_hash):
            self._compiled_gradient_fn = self._compile_gradient_function(params)
            self._last_params_hash = params_hash
        
        return self._compiled_gradient_fn(eta)
    
    def compute_hessian(self, 
                       params: Union[Dict, FrozenDict], 
                       eta: jnp.ndarray) -> jnp.ndarray:
        """
        Compute Hessian of the log normalizer (covariance of sufficient statistics).
        
        Args:
            params: Model parameters
            eta: Natural parameters [batch_size, eta_dim]
            
        Returns:
            Hessians [batch_size, eta_dim] (diagonal) or [batch_size, eta_dim, eta_dim] (full)
        """
        # Check if we need to recompile functions
        params_hash = self._get_params_hash(params)
        if (self._compiled_hessian_fn is None or 
            self._last_params_hash != params_hash):
            self._compiled_hessian_fn = self._compile_hessian_function(params)
            self._last_params_hash = params_hash
        
        return self._compiled_hessian_fn(eta)
    
    def compute_gradient_and_hessian(self, 
                                   params: Union[Dict, FrozenDict], 
                                   eta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute both gradient and Hessian in a single call for efficiency.
        
        Args:
            params: Model parameters
            eta: Natural parameters [batch_size, eta_dim]
            
        Returns:
            Tuple of (gradients, hessians)
        """
        gradient = self.compute_gradient(params, eta)
        hessian = self.compute_hessian(params, eta)
        return gradient, hessian


class LogNormalizerDerivatives:
    """
    Specialized class for computing derivatives of log normalizers in exponential families.
    
    This class provides optimized functions for the specific use case of computing
    E[T(X)|eta] (gradient) and Cov[T(X)|eta] (Hessian) for exponential family distributions.
    """
    
    def __init__(self, 
                 model_apply_fn: Callable,
                 hessian_method: str = 'full',
                 compile_functions: bool = True):
        """
        Initialize the log normalizer derivatives utilities.
        
        Args:
            model_apply_fn: The model's apply function
            hessian_method: Method for Hessian computation ('diagonal' or 'full')
            compile_functions: Whether to JIT compile the functions
        """
        self.utils = GradientHessianUtils(
            model_apply_fn=model_apply_fn,
            hessian_method=hessian_method,
            compile_functions=compile_functions
        )
    
    def predict_mean(self, 
                    params: Union[Dict, FrozenDict], 
                    eta: jnp.ndarray) -> jnp.ndarray:
        """
        Predict mean sufficient statistics E[T(X)|eta] using gradient of log normalizer.
        
        Args:
            params: Model parameters
            eta: Natural parameters [batch_size, eta_dim]
            
        Returns:
            Predicted means [batch_size, eta_dim]
        """
        return self.utils.compute_gradient(params, eta)
    
    def predict_covariance(self, 
                          params: Union[Dict, FrozenDict], 
                          eta: jnp.ndarray) -> jnp.ndarray:
        """
        Predict covariance of sufficient statistics Cov[T(X)|eta] using Hessian of log normalizer.
        
        Args:
            params: Model parameters
            eta: Natural parameters [batch_size, eta_dim]
            
        Returns:
            Predicted covariances with shape:
            - Diagonal method: [batch_size, eta_dim] - diagonal elements only
            - Full method: [batch_size, eta_dim, eta_dim] - full covariance matrix
            
        Note:
            The diagonal method returns only the diagonal elements of the Hessian matrix,
            which represent the variance of each eta dimension independently.
            The full method returns the complete Hessian matrix for full covariance information.
        """
        return self.utils.compute_hessian(params, eta)
    
    def get_hessian_method(self) -> str:
        """Get the current Hessian computation method."""
        return self.utils.hessian_method
    
    def predict_mean_and_covariance(self, 
                                  params: Union[Dict, FrozenDict], 
                                  eta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predict both mean and covariance in a single call.
        
        Args:
            params: Model parameters
            eta: Natural parameters [batch_size, eta_dim]
            
        Returns:
            Tuple of (predicted_means, predicted_covariances)
        """
        return self.utils.compute_gradient_and_hessian(params, eta)


# Convenience functions for direct use
def create_gradient_utils(model_apply_fn: Callable, 
                        hessian_method: str = 'full',
                        compile_functions: bool = True) -> GradientHessianUtils:
    """
    Create gradient/hessian utilities for the given model.
    
    Args:
        model_apply_fn: The model's apply function
        hessian_method: Method for Hessian computation ('diagonal' or 'full')
        compile_functions: Whether to JIT compile the functions
        
    Returns:
        GradientHessianUtils instance
    """
    return GradientHessianUtils(
        model_apply_fn=model_apply_fn,
        hessian_method=hessian_method,
        compile_functions=compile_functions
    )


def create_log_normalizer_derivatives(model_apply_fn: Callable, 
                                    hessian_method: str = 'full',
                                    compile_functions: bool = True) -> LogNormalizerDerivatives:
    """
    Create log normalizer derivatives utilities for the given model.
    
    Args:
        model_apply_fn: The model's apply function
        hessian_method: Method for Hessian computation ('diagonal' or 'full')
        compile_functions: Whether to JIT compile the functions
        
    Returns:
        LogNormalizerDerivatives instance
    """
    return LogNormalizerDerivatives(
        model_apply_fn=model_apply_fn,
        hessian_method=hessian_method,
        compile_functions=compile_functions
    )


# Legacy compatibility functions
def compile_gradient_function(model_apply_fn: Callable, 
                            params: Union[Dict, FrozenDict]) -> Callable:
    """
    Legacy function for compiling gradient function.
    
    Args:
        model_apply_fn: The model's apply function
        params: Model parameters
        
    Returns:
        Compiled gradient function
    """
    utils = GradientHessianUtils(model_apply_fn, compile_functions=True)
    return utils._compile_gradient_function(params)


def compile_hessian_function(model_apply_fn: Callable, 
                           params: Union[Dict, FrozenDict],
                           hessian_method: str = 'full') -> Callable:
    """
    Legacy function for compiling Hessian function.
    
    Args:
        model_apply_fn: The model's apply function
        params: Model parameters
        hessian_method: Method for Hessian computation ('diagonal' or 'full')
        
    Returns:
        Compiled Hessian function
    """
    utils = GradientHessianUtils(model_apply_fn, hessian_method=hessian_method, compile_functions=True)
    return utils._compile_hessian_function(params)
