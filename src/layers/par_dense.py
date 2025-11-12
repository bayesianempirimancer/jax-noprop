# Implements a parallel dense layer that takes input with nontrivial shape (..., in_feat_m2, in_feat_m1) and outputs (..., in_feat_m2, out_feat_m1)
# by applying a different dense operation on for each in_feat_m2 index

import jax.numpy as jnp
import flax.linen as nn

class ParDense(nn.Module):
    """Parallel dense layer that takes input with nontrivial shape (..., in_feat_m2, in_feat_m1) and outputs (..., in_feat_m2, out_feat_m1)
    by applying a different dense operation on for each in_feat_m2 index"""
    in_feat_m2: int
    in_feat_m1: int
    out_feat_m1: int

    def setup(self):
        kernel = self.param('kernel', nn.initializers.normal(), (self.in_feat_m2, self.in_feat_m1, self.out_feat_m1))
        bias = self.param('bias', nn.initializers.zeros, (self.in_feat_m2, self.out_feat_m1))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.einsum('...ij,ijk->...ik', x, self.kernel) + self.bias
    