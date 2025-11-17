"""
Flow Planner

The purpose of the flow planner is to generate the various quantities that are needed to train a flow model.  It is meant to
encapsulate both the logic of the noise scheduling and the logic of the transport plan.  Now in an optimal transport setting, 
a flow planner would take in samples from the target and initial distributions and generate a transport plan by identifying 
which samples should be mapped onto which other samples.  This is usually accomplished using the sinkhorn algorithm or by 
actually solving the optimal transport problem.  Both of these approaches are fairly computationally expensive and not super 
flexible as they require a commitment to a particular distribution for the initial conditions of the flow.  

Here we consider a different approach.  Rather than solving the optimal transport problem, we consider the problem of 
intelligently selecting the distribution of the initial conditions for the flow given a mini-batch of samples from the target
distribution.  This amounts to selecting a parameterized plan pi(x_0 | x_1).  

For simplicity the plan we will choose will be a Gaussian mixture model with shared means (and optionally shared variances).  The 
basic idea is that we will use the vbgmm code in the vae model to coarsely cluster the samples from the target distribution and 
extract assignment variables for each sample.  We will then construct a transport plan by sampling from the top k clusters associated
with each data point (x_1).  These samples will then be used as the initial conditions for the flow model.  

Thus the principle output of this module will be an x_0 that has the same shape as x_target (or x_1).  Model parameters will be 
stored internally.  We will also need an update routine to learn the parameters of the gmm that we can implement as part of an 
external train step routine.

"""

import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
from flax.core import FrozenDict, freeze, unfreeze
from typing import Tuple, Optional, Dict, Any
from functools import partial
from dataclasses import field, MISSING

from src.vae.vb_gmm import GMMVBEM, GMMVBEMConfig, create_gmm_vbem
from src.utils.math_utils import stable_softmax
from src.embeddings.flow_schedules import FlowSchedule, FlowScheduleConfig, create_flow_schedule, LinearFlowSchedule


class FlowPlanner(LinearFlowSchedule):
    """Gaussian Mixture Flow Planner for generating initial conditions x_0 from target samples x_1.
    
    This planner inherits from FlowSchedule (specifically LinearFlowSchedule by default) and adds
    GMM-based methods for generating initial conditions x_0 from target samples x_1.
    
    The planner uses a GMM to cluster target samples and generates initial conditions by sampling
    from the top-k clusters associated with each target sample.
    
    The GMM uses shared means across clusters (all clusters share the same mean) and optionally
    shared variances. This simplifies the model while still allowing for flexible transport plans.
    """
    # GMM configuration (optional in field definition, but required in practice)
    gmm_config: GMMVBEMConfig = None  # GMM configuration
    # Optional fields (after required fields)
    top_k: int = 3  # Number of top clusters to sample from for each data point
    
    def setup(self):
        """Initialize the GMM component."""
        
        # Validate gmm_config is provided
        if self.gmm_config is None:
            raise ValueError("gmm_config must be provided when creating FlowPlanner")
        
        # Create GMM using factory function
        self.gmm = create_gmm_vbem(self.gmm_config)
    
    def __call__(self, x_target: jnp.ndarray, key: jr.PRNGKey, training: bool = True) -> jnp.ndarray:
        """
        Initialize all nn.compact methods.
        
        This is called by Flax's init() to initialize all @nn.compact methods.
        The GMM cluster means should be initialized separately using initialize()
        after init() is called.
        
        Args:
            x_target: Target samples [batch, ..., latent_dim] (used for shape inference)
            key: Random key for sampling
            training: Whether in training mode
            
        Returns:
            Dummy output for initialization
        """
        # Initialize compact methods by calling them
        # Use dummy data to initialize the GMM's compact methods
        batch_shape = x_target.shape[:-1]
        _ = self.gmm.sample(key=key, batch_shape=batch_shape, training=training)
        
        # Initialize flow schedule's compact methods (inherited from FlowSchedule)
        # Use dummy time values
        t_dummy = jnp.zeros(batch_shape) if len(batch_shape) > 0 else jnp.array(0.5)
        x_0_dummy = jnp.zeros(x_target.shape)
        _ = self.x_t(x_0_dummy, x_target, t_dummy)
        
        return jnp.zeros(x_target.shape)
    
    def initialize_gmm(self, params: dict, x_target: jnp.ndarray, key: jr.PRNGKey) -> dict:
        """
        Initialize GMM cluster means from target data.
        
        This should be called after model.init() to set the cluster means based on actual data.
        Uses the GMM's get_initial_cluster_means method and updates params using JAX tree utilities.
        
        Args:
            params: Model parameters (from model.init())
            x_target: Target samples [batch, ..., latent_dim] or [N, latent_dim]
            key: Random key for sampling
            
        Returns:
            Updated params dictionary with initialized cluster means
        """
        import jax.tree_util as jtu
        
        # Verify GMM params exist
        gmm_params = params.get('params', {}).get('gmm', {})
        if 'mu_n' not in gmm_params:
            raise ValueError("GMM params not initialized. Please call model.init() first.")
        
        # Call GMM's get_initial_cluster_means class method to get mu_n
        mu_n = GMMVBEM.get_initial_cluster_means(
            num_clusters=self.gmm_config.num_clusters,
            latent_dim=self.gmm_config.latent_dim,
            x=x_target,
            key=key
        )
        
        # Update gmm_params with initialized cluster means
        gmm_params['mu_n'] = mu_n
        updated_gmm_params = gmm_params
        
        # Update params using JAX tree utilities for robust nested updates
        def update_gmm_params(path, value):
            # Check if we're at the gmm submodule
            if len(path) >= 2 and path[-2:] == ('params', 'gmm'):
                return updated_gmm_params
            return value
        
        params_updated = jtu.tree_map_with_path(update_gmm_params, params)
        
        return params_updated


    @nn.compact
    def sample_x_0(
        self, 
        x_target: jnp.ndarray, 
        key: jr.PRNGKey, 
        method: str = "mixture",
        training: bool = True
    ) -> jnp.ndarray:
        """
        Generate initial conditions x_0 from target samples x_1.
        
        Args:
            x_target: Target samples [batch, ..., latent_dim] (x_1)
            key: Random key for sampling
            method: Sampling method - "mixture" (GMM conditional sampling) or "normal" (standard normal)
            training: Whether in training mode
            
        Returns:
            x_0: Initial conditions [batch, ..., latent_dim] with same shape as x_target
        """
        if method == "mixture":
            # Use GMM's conditional sample method to generate samples based on target
            x_0 = self.gmm.sample_conditional(
                x=x_target,
                key=key,
                top_k=self.top_k,
                training=training
            )
        elif method == "normal":
            # Sample from standard normal distribution
            x_0 = jr.normal(key, x_target.shape)
        else:
            raise ValueError(f"Unknown sampling method: {method}. Must be 'mixture' or 'normal'.")
        
        return x_0
    





def create_flow_planner(
    config_dict: Dict[str, Any],
    latent_shape: Tuple[int, ...],
    input_shape: Tuple[int, ...],
    output_shape: Optional[Tuple[int, ...]] = None
) -> nn.Module:
    """
    Factory function to create a FlowPlanner instance.
    
    Args:
        config_dict: Configuration dictionary with flow planner settings
        latent_shape: Shape of latent space (should match x_target shape)
        input_shape: Shape of input (not used for flow planner, but kept for consistency)
        output_shape: Shape of output (not used for flow planner, but kept for consistency)
        
    Returns:
        FlowPlanner instance
    """
    # Extract latent dimension from latent_shape
    if isinstance(latent_shape, tuple) and len(latent_shape) > 0:
        latent_dim = int(jnp.prod(jnp.array(latent_shape)))
    else:
        latent_dim = int(latent_shape) if isinstance(latent_shape, (int, float)) else 1
    
    # Get configuration values with defaults
    num_clusters = config_dict.get("num_clusters", 512)
    top_k = config_dict.get("top_k", 1)
    shared_variances = config_dict.get("shared_variances", False)
    prior_mu = config_dict.get("prior_mu", 0.0)
    prior_alpha = config_dict.get("prior_alpha", 1.0)
    prior_beta = config_dict.get("prior_beta", 1.0)
    prior_alpha_mix = config_dict.get("prior_alpha_mix", 0.5)
    beta_mix = config_dict.get("beta_mix", 0.1)
    
    # Create GMM config
    gmm_config = GMMVBEMConfig(
        num_clusters=num_clusters,
        latent_dim=latent_dim,
        prior_mu=prior_mu,
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        prior_alpha_mix=prior_alpha_mix,
        beta_mix=beta_mix,
        tie_precisions=shared_variances  # Map shared_variances to tie_precisions
    )
    
    # Get flow schedule parameters (FlowPlanner inherits from LinearFlowSchedule)
    # Extract schedule parameters from config_dict or use defaults
    flow_schedule_config = config_dict.get("flow_schedule_config")
    if flow_schedule_config is None:
        # Default to linear flow schedule parameters
        alpha_min = config_dict.get("alpha_min", 0.05)
        alpha_max = config_dict.get("alpha_max", 0.95)
        sigma_min = config_dict.get("sigma_min", 0.05)
        sigma_max = config_dict.get("sigma_max", 0.95)
        learnable = config_dict.get("learnable", False)
    else:
        # Extract from FlowScheduleConfig if provided
        if isinstance(flow_schedule_config, FlowScheduleConfig):
            alpha_min = flow_schedule_config.alpha_min
            alpha_max = flow_schedule_config.alpha_max
            sigma_min = flow_schedule_config.sigma_min
            sigma_max = flow_schedule_config.sigma_max
            learnable = flow_schedule_config.learnable
        else:
            # Assume it's a dict
            alpha_min = flow_schedule_config.get("alpha_min", 0.05)
            alpha_max = flow_schedule_config.get("alpha_max", 0.95)
            sigma_min = flow_schedule_config.get("sigma_min", 0.05)
            sigma_max = flow_schedule_config.get("sigma_max", 0.95)
            learnable = flow_schedule_config.get("learnable", False)
    
    return FlowPlanner(
        ndims=1,  # Number of dimensions in x_shape (latent_dim is a single dimension)
        learnable=learnable,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        gmm_config=gmm_config,
        top_k=top_k
    )
