from flax.linen import dot_product_attention, DenseGeneral, LayerNorm
import flax.linen as nn
import jax.numpy as jnp
from typing import Any, Callable, Sequence, Optional, Union

class MAB(nn.Module):
    """ Implementation of the 'Multihead Attention Block'.

    Parameters
    ----------
    N_dim : int
        element-wise size of the output 

    N_head : int
        number of attention heads, must be a divisor of N_dim

    ln : bool
        if set to False, there is no layer normalization applied
        default: False
    """

    N_dim: int
    N_head: int
    ln: bool = False
        
    @nn.compact
    def __call__(self, x, y):
        N_batch = x.shape[0]
        N_elements = x.shape[1]
        dim_split = self.N_dim//self.N_head
        kernel_init = nn.initializers.variance_scaling(scale = 1/3, mode = "fan_in", distribution = "uniform")
        
        q = DenseGeneral(features = (self.N_head, dim_split), kernel_init = kernel_init)(x)
        k = DenseGeneral(features = (self.N_head, dim_split), kernel_init = kernel_init)(y)
        v = DenseGeneral(features = (self.N_head, dim_split), kernel_init = kernel_init)(y)
        
        a = nn.dot_product_attention(q,k,v)

        o = (a+q).reshape(N_batch, N_elements, self.N_dim)
        
        if self.ln:
            o = LayerNorm()(o)
            
        o = o + nn.activation.relu(DenseGeneral(self.N_dim, kernel_init = kernel_init)(o))
        if self.ln:
            o = LayerNorm()(o)
        
        return o
    
class SAB(nn.Module):
    """ Implementation of the 'Set Attention Block'.

    Parameters
    ----------
    N_dim : int
        element-wise size of the output 

    N_head : int
        number of attention heads, must be a divisor of N_dim

    ln : bool
        if set to False, there is no layer normalization applied
        default: False
    """
    N_dim: int
    N_head: int
    ln: bool = False
        
    @nn.compact
    def __call__(self, x):
        return MAB(N_dim = self.N_dim,
                  N_head = self.N_head,
                  ln = self.ln)(x,x)

class ISAB(nn.Module):
    """ Implementation of the 'Induced Set Attention Block'.

    Parameters
    ----------
    N_dim : int
        element-wise size of the output 

    N_head : int
        number of attention heads, must be a divisor of N_dim
        
    N_induced: int
        number of 'inducing points'

    ln : bool
        if set to False, there is no layer normalization applied
        default: False
    """
    N_dim: int
    N_head: int
    N_induced: int
    ln: bool = False
        
    @nn.compact
    def __call__(self, x):
        N_batch = x.shape[0]
        i = self.param("induced_points", nn.initializers.xavier_uniform(), (1,self.N_induced, self.N_dim))
        i = jnp.repeat(i, N_batch,axis=0).reshape((N_batch,self.N_induced,self.N_dim))
        
        h = MAB(N_dim = self.N_dim,
                N_head = self.N_head,
                ln = self.ln) (i, x)
        
        return MAB(N_dim = self.N_dim,
                   N_head = self.N_head,
                   ln = self.ln) (x, h)
    
class PMA(nn.Module):
    """ Implementation of the 'Pooling by Multihead Attention'.

    Parameters
    ----------
    N_dim : int
        element-wise size of the output 

    N_head : int
        number of attention heads, must be a divisor of N_dim
        
    N_seed: int
        number of 'seed vectors'

    ln : bool
        if set to False, there is no layer normalization applied
        default: False
    """
    N_dim: int
    N_head: int
    N_seed: int
    ln: bool = False
        
    @nn.compact
    def __call__(self, x):
        N_batch = x.shape[0]
        s = self.param("seeds", nn.initializers.xavier_uniform(), (1,self.N_seed,self.N_dim))
        s = jnp.repeat(s, N_batch,axis=0).reshape((N_batch,self.N_seed,self.N_dim))
        
        return MAB(N_dim = self.N_dim,
                   N_head = self.N_head,
                   ln = self.ln) (s, x)