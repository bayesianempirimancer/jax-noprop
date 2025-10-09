"""JAX NoProp: A Jax Implementation of the NoProp Algorithm

This package provides implementations of the NoProp (No Propagation) algorithm
for training neural networks without backpropagation. It includes three variations:
- noprop-dt: Discrete time version
- noprop-ct: Continuous time version
- noprop-fm: Flow matching version
"""

from jax_noprop.noprop_dt import NoPropDT
from jax_noprop.noprop_ct import NoPropCT
from jax_noprop.noprop_fm import NoPropFM

__version__ = "0.1.0"
__all__ = ["NoPropDT", "NoPropCT", "NoPropFM"]
