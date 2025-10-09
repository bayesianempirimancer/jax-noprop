# Copilot Instructions for jax-noprop

## Project Overview
This repository contains a JAX implementation of the NoProp algorithm. JAX is a Python library for high-performance numerical computing and machine learning.

## Code Style and Standards
- Follow PEP 8 Python style guidelines
- Use type hints for function signatures
- Write docstrings in NumPy/Google style format
- Keep functions pure and functional where possible (following JAX best practices)

## JAX-Specific Guidelines
- Use JAX's functional approach: functions should be pure (no side effects)
- Use `jax.numpy` instead of regular NumPy where applicable
- Leverage JAX transformations (`jit`, `grad`, `vmap`, `pmap`) appropriately
- Ensure code is compatible with JAX's JIT compilation constraints
- Avoid in-place array modifications (JAX arrays are immutable)
- Use `jax.random.PRNGKey` for random number generation

## Development Practices
- Write clear, self-documenting code with meaningful variable names
- Add comments only when the code logic is complex or non-obvious
- Optimize for readability first, then performance
- When implementing algorithms, include references to papers or documentation

## Testing
- Write unit tests for new functionality
- Ensure tests are deterministic (use fixed random seeds when needed)
- Test both forward and backward passes for gradient-based operations

## Dependencies
- Use JAX as the primary numerical computing library
- Minimize external dependencies when possible
- Document any new dependencies added to the project
