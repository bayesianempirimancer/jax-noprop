#!/usr/bin/env python
"""JAX-compatible matrix utility functions for upper triangular operations."""

import jax.numpy as jnp
from typing import Union

Array = Union[jnp.ndarray]

def vector_to_upper_triangular(vector: Array, n: int) -> Array:
    """
    Convert vectors to upper triangular matrices including diagonal.
    Works with arbitrary batch shapes.
    
    Args:
        vector: Array with shape (..., n*(n+1)/2) containing upper triangular elements
        n: Size of the resulting n×n matrix
        
    Returns:
        Array with shape (..., n, n) - upper triangular matrices with zeros in lower part
        
    Example:
        >>> # Single vector
        >>> vector = jnp.array([1, 2, 3, 4, 5, 6])  # n=3, length = 3*(3+1)/2 = 6
        >>> matrix = vector_to_upper_triangular(vector, 3)
        >>> print(matrix.shape)  # (3, 3)
        
        >>> # Batch of vectors
        >>> batch_vectors = jnp.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
        >>> batch_matrices = vector_to_upper_triangular(batch_vectors, 3)
        >>> print(batch_matrices.shape)  # (2, 3, 3)
    """
    # Get batch shape and validate input
    batch_shape = vector.shape[:-1]
    expected_vec_size = n * (n + 1) // 2
    assert vector.shape[-1] == expected_vec_size, f"Expected vector size {expected_vec_size}, got {vector.shape[-1]}"
    
    # Create upper triangular mask
    upper_mask = jnp.triu(jnp.ones((n, n)), k=0)
    upper_indices = jnp.where(upper_mask.flatten())[0]
    
    # Create output array with batch dimensions
    output_shape = batch_shape + (n, n)
    matrices = jnp.zeros(output_shape)
    
    # Flatten batch dimensions for easier indexing
    flat_matrices = matrices.reshape(-1, n * n)
    flat_vectors = vector.reshape(-1, expected_vec_size)
    
    # Set upper triangular elements for all matrices in the batch
    flat_matrices = flat_matrices.at[:, upper_indices].set(flat_vectors)
    
    # Reshape back to original batch shape + (n, n)
    return flat_matrices.reshape(output_shape)

def upper_triangular_to_vector(matrix: Array) -> Array:
    """
    Extract upper triangular elements (including diagonal) from matrices into vectors.
    Works with arbitrary batch shapes.
    
    Args:
        matrix: Array with shape (..., n, n) - matrices
        
    Returns:
        Array with shape (..., n*(n+1)/2) containing upper triangular elements
        
    Example:
        >>> # Single matrix
        >>> matrix = jnp.array([[1, 2, 3], [0, 4, 5], [0, 0, 6]])
        >>> vector = upper_triangular_to_vector(matrix)
        >>> print(vector.shape)  # (6,)
        
        >>> # Batch of matrices
        >>> batch_matrices = jnp.zeros((2, 3, 3))  # 2 matrices of size 3x3
        >>> batch_vectors = upper_triangular_to_vector(batch_matrices)
        >>> print(batch_vectors.shape)  # (2, 6)
    """
    # Get dimensions
    batch_shape = matrix.shape[:-2]
    n = matrix.shape[-1]
    assert matrix.shape[-2] == n, f"Expected square matrices, got shape {matrix.shape}"
    
    # Create upper triangular mask
    upper_mask = jnp.triu(jnp.ones((n, n)), k=0)
    upper_indices = jnp.where(upper_mask.flatten())[0]
    
    # Flatten batch dimensions for easier indexing
    flat_matrices = matrix.reshape(-1, n * n)
    
    # Extract upper triangular elements
    flat_vectors = flat_matrices[:, upper_indices]
    
    # Reshape back to original batch shape + (vec_size,)
    vec_size = n * (n + 1) // 2
    output_shape = batch_shape + (vec_size,)
    return flat_vectors.reshape(output_shape)

def test_functions():
    """Test the functions with both single matrices and batched operations."""
    print("Testing upper triangular functions with arbitrary batch shapes...")
    
    # Test case 1: Single 3x3 matrix
    print("\n=== Single Matrix Test ===")
    vector_3d = jnp.array([1, 2, 3, 4, 5, 6])  # 3*(3+1)/2 = 6 elements
    n_3d = 3
    
    matrix_3d = vector_to_upper_triangular(vector_3d, n_3d)
    print(f"Input vector shape: {vector_3d.shape}")
    print(f"Output matrix shape: {matrix_3d.shape}")
    print(f"Output matrix:\n{matrix_3d}")
    
    vector_back = upper_triangular_to_vector(matrix_3d)
    print(f"Extracted vector shape: {vector_back.shape}")
    print(f"Round-trip successful: {jnp.allclose(vector_3d, vector_back)}")
    
    # Test case 2: 1D Batch (3 matrices)
    print("\n=== 1D Batch Test ===")
    batch_vectors_1d = jnp.array([
        [1, 2, 3, 4, 5, 6],      # First matrix
        [7, 8, 9, 10, 11, 12],   # Second matrix
        [13, 14, 15, 16, 17, 18] # Third matrix
    ])  # Shape: (3, 6)
    
    batch_matrices_1d = vector_to_upper_triangular(batch_vectors_1d, 3)
    print(f"Batch vectors shape: {batch_vectors_1d.shape}")
    print(f"Batch matrices shape: {batch_matrices_1d.shape}")
    print(f"First matrix:\n{batch_matrices_1d[0]}")
    
    batch_vectors_back_1d = upper_triangular_to_vector(batch_matrices_1d)
    print(f"Round-trip successful: {jnp.allclose(batch_vectors_1d, batch_vectors_back_1d)}")
    
    # Test case 3: 2D Batch (2x2 grid of matrices)
    print("\n=== 2D Batch Test ===")
    batch_vectors_2d = jnp.array([
        [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],     # First row
        [[13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]] # Second row
    ])  # Shape: (2, 2, 6)
    
    batch_matrices_2d = vector_to_upper_triangular(batch_vectors_2d, 3)
    print(f"2D batch vectors shape: {batch_vectors_2d.shape}")
    print(f"2D batch matrices shape: {batch_matrices_2d.shape}")
    print(f"Matrix at [0,0]:\n{batch_matrices_2d[0, 0]}")
    print(f"Matrix at [1,1]:\n{batch_matrices_2d[1, 1]}")
    
    batch_vectors_back_2d = upper_triangular_to_vector(batch_matrices_2d)
    print(f"2D round-trip successful: {jnp.allclose(batch_vectors_2d, batch_vectors_back_2d)}")
    
    # Test case 4: 3D Batch (arbitrary shape)
    print("\n=== 3D Batch Test ===")
    batch_vectors_3d = jnp.ones((2, 3, 4, 6))  # Shape: (2, 3, 4, 6) - 24 matrices total
    
    batch_matrices_3d = vector_to_upper_triangular(batch_vectors_3d, 3)
    print(f"3D batch vectors shape: {batch_vectors_3d.shape}")
    print(f"3D batch matrices shape: {batch_matrices_3d.shape}")
    
    batch_vectors_back_3d = upper_triangular_to_vector(batch_matrices_3d)
    print(f"3D round-trip successful: {jnp.allclose(batch_vectors_3d, batch_vectors_back_3d)}")
    
    # Test case 5: Different matrix sizes
    print("\n=== Different Matrix Sizes Test ===")
    for n in [2, 3, 4, 5]:
        vec_size = n * (n + 1) // 2
        test_vector = jnp.arange(1, vec_size + 1, dtype=jnp.float32)
        test_matrix = vector_to_upper_triangular(test_vector, n)
        test_vector_back = upper_triangular_to_vector(test_matrix)
        success = jnp.allclose(test_vector, test_vector_back)
        print(f"n={n}, vec_size={vec_size}, round-trip: {success}")

if __name__ == "__main__":
    test_functions()
