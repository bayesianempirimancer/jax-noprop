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


class GMMFlowPlanner(LinearFlowSchedule):
    """Gaussian Mixture Flow Planner for generating initial conditions x_0 from target samples x_1.
    
    This planner inherits from FlowSchedule (specifically LinearFlowSchedule by default) and adds
    GMM-based methods for generating initial conditions x_0 from target samples x_1.
    
    The planner uses a GMM to cluster target samples and generates initial conditions by sampling
    from the top-k clusters associated with each target sample.
    """

    gmm_config: GMMVBEMConfig
    top_k: int = 3  # Number of top clusters to sample from for each data point
    sample_method: str = "mixture"  # "mixture" or "normal"
    sinkhorn_refinement: bool = False  # Whether to use sinkhorn refinement (default: False)
    
    def setup(self):
        """Initialize the GMM component."""
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
    
    def fit_gmm(
        self, 
        params: dict, 
        x_data: jnp.ndarray, 
        initialize: bool = False,
        num_epochs: int = 10,
        batch_size: int = 256,
        N_eff: Optional[float] = None,
        lr: float = 0.2,
        seed: int = 42
    ) -> dict:
        """
        Fit GMM to data using VBEM updates.
        
        This method fits the GMM component of the flow planner to the provided data.
        It handles initialization and VBEM updates to learn the GMM parameters.
        
        Args:
            params: Model parameters (from model.init())
            x_data: Training data [N, latent_dim] or [batch, ..., latent_dim]
            initialize: If True, initialize cluster means from data before fitting
            num_epochs: Number of training epochs
            batch_size: Batch size for VBEM updates
            N_eff: Effective number of data points (if None, uses x_data.shape[0])
            lr: Learning rate for VBEM updates (mixing parameter between 0 and 1)
            seed: Random seed for shuffling data and initialization
            
        Returns:
            Updated params dictionary with fitted GMM parameters
        """
        return GMMVBEM.fit(
            config=self.gmm_config,
            params=params,
            x_data=x_data,
            apply_fn=self.apply,  # Pass planner's apply method
            initialize=initialize,
            num_epochs=num_epochs,
            batch_size=batch_size,
            N_eff=N_eff,
            lr=lr,
            seed=seed
        )
    
    def gmm_update(
        self,
        params: dict,
        x_batch: jnp.ndarray,
        N_eff: float = 2000.0,
        lr: float = 0.2,
        training: bool = True
    ) -> dict:
        """
        Update GMM parameters using VBEM on a single minibatch.
        
        This method performs a single VBEM update step on a minibatch of data.
        Useful for incremental updates during training.
        
        Args:
            params: Model parameters (from model.init())
            x_batch: Minibatch of training data [batch_size, latent_dim]
            N_eff: Effective number of data points (inverse temperature)
            lr: Learning rate for VBEM updates (mixing parameter between 0 and 1)
            training: Whether in training mode
            
        Returns:
            Updated params dictionary with updated GMM parameters
        """
        from flax.core import unfreeze, freeze
        
        # Call GMM's update method via planner's apply
        updated_params_dict = self.apply(
            params,
            x_batch,
            N_eff=N_eff,
            lr=lr,
            training=training,
            method=lambda mdl, x, **kwargs: mdl.gmm.update(x, **kwargs)
        )
        
        # Update params structure
        params_unfrozen = unfreeze(params)
        gmm_params = params_unfrozen.get('params', {}).get('gmm', {})
        
        # Update gmm_params with the returned values
        gmm_params = {
            'mu_n': updated_params_dict['mu_n'],
            'alpha_n': updated_params_dict['alpha_n'],
            'beta_n': updated_params_dict['beta_n'],
            'alpha_mix': updated_params_dict['alpha_mix']
        }
        
        # Update the nested structure
        params_unfrozen['params']['gmm'] = gmm_params
        
        # Freeze and return
        return freeze(params_unfrozen)

    @nn.compact
    def sample_x_0(
        self, 
        x_target: jnp.ndarray, 
        key: jr.PRNGKey, 
        training: bool = True
    ) -> jnp.ndarray:
        """
        Generate initial conditions x_0 unconditionally.
        
        Uses the sampling method specified in self.sample_method (set via config):
        - "mixture": GMM unconditional sampling (samples from GMM prior, not conditioned on x_target)
        - "normal": Standard normal distribution
        
        Args:
            x_target: Target samples [batch, ..., latent_dim] (x_1) - used only for shape and Sinkhorn refinement, not for conditioning
            key: Random key for sampling
            training: Whether in training mode
            
        Returns:
            x_0: Initial conditions [batch, ..., latent_dim] with same shape as x_target
        """
        key, sample_key, noise_key = jr.split(key, 3)
        if self.sample_method == "mixture":
            # Use GMM's unconditional sample method (sample from GMM prior, not conditioned on x_target)
            # Extract batch_shape from x_target
            batch_shape = x_target.shape[:-1]  # Remove latent_dim dimension
            # Ensure batch_shape is not empty (handle case where x_target is 1D)
            if len(batch_shape) == 0:
                batch_shape = (1,)
            x_0 = self.gmm.sample(
                key=sample_key,
                batch_shape=batch_shape,
                training=training
            )
#            x_0 = x_0 + jr.normal(noise_key, x_0.shape)
        elif self.sample_method == "normal":
            # Sample from standard normal distribution
            x_0 = jr.normal(noise_key, x_target.shape)
        else:
            raise ValueError(f"Unknown sampling method: {self.sample_method}. Must be 'mixture' or 'normal'.")
        
        # Apply sinkhorn refinement if enabled and in training mode (not during prediction)
        # Sinkhorn refinement requires x_target, which we only have during training
        # Skip sinkhorn refinement during initialization (when x_target might be 1D)
        if self.sinkhorn_refinement and training and x_target.ndim >= 2:
            key, sinkhorn_key = jr.split(key)
            x_0 = self.apply_sinkhorn_refinement(x_0, x_target, sinkhorn_key, training=training)
        
        return x_0

    @nn.compact
    def compute_gmm_loss(
        self,
        x_target: jnp.ndarray,
        training: bool = True
    ) -> jnp.ndarray:
        """
        Compute GMM loss for x_target without generating samples.
        
        This computes the GMM loss based on x_target, which represents
        the negative log-likelihood plus KL divergence.
        
        Args:
            x_target: Target samples [batch, ..., latent_dim] to compute loss for
            training: Whether in training mode
            
        Returns:
            gmm_loss: GMM loss value [scalar] - loss for mixture, 0 for normal
        """
        if self.sample_method == "mixture":
            # Flatten x_target to [N, latent_dim] for GMM loss computation
            original_shape = x_target.shape
            x_target_flat = x_target.reshape(-1, original_shape[-1])  # [N, latent_dim]
            
            # Compute GMM loss directly
            gmm_loss = self.gmm.loss(x_target_flat, training=training)
            return gmm_loss
        else:
            # For normal sampling, no GMM loss
            return jnp.array(0.0)

    @nn.compact
    def apply_sinkhorn_refinement(
        self, 
        x_0: jnp.ndarray, 
        x_target: jnp.ndarray, 
        key: jr.PRNGKey, 
        training: bool = True,
        epsilon: float = 0.1,
        num_iterations: int = 100
    ) -> jnp.ndarray:
        """
        Refines current x_0 using the Sinkhorn algorithm to optimally align x_0 with a target distribution.
        
        This function uses optimal transport theory to find a reordering of x_0 that minimizes the 
        transport cost to x_target. The Sinkhorn algorithm solves the entropic regularized optimal 
        transport problem.
        
        Args:
            x_0: Initial conditions [batch, ..., latent_dim] to be reordered
            x_target: Target samples [batch, ..., latent_dim] to align with
            key: Random key for sampling (not used in deterministic Sinkhorn, kept for interface consistency)
            training: Whether in training mode (not used, kept for interface consistency)
            epsilon: Entropic regularization parameter (smaller = more precise but less stable)
            num_iterations: Number of Sinkhorn iterations
            
        Returns:
            x_0_reordered: Reordered x_0 [batch, ..., latent_dim] aligned with x_target distribution
        """
        # Flatten all dimensions except the last (latent_dim) into batch dimension
        original_shape = x_0.shape
        latent_dim = original_shape[-1]
        
        # Reshape to [batch_size, num_samples, latent_dim]
        # Flatten all leading dimensions into a single batch dimension
        # For shape [d1, d2, ..., dN, latent_dim], we treat [d1, ..., dN-1] as batch
        # and dN as num_samples, OR if only 2 dims, treat as [num_samples, latent_dim]
        if len(original_shape) == 2:
            # Unbatched case: [num_samples, latent_dim]
            x_0_flat = x_0[None, :, :]  # [1, num_samples, latent_dim]
            x_target_flat = x_target[None, :, :]  # [1, num_samples, latent_dim]
            batch_size = 1
            num_samples = original_shape[0]
        else:
            # Batched case: [..., num_samples, latent_dim] or [batch, num_samples, latent_dim]
            # Flatten all but last two dimensions into batch, keep last two as [num_samples, latent_dim]
            batch_size = int(jnp.prod(jnp.array(original_shape[:-2]))) if len(original_shape) > 2 else 1
            num_samples = original_shape[-2] if len(original_shape) > 2 else original_shape[0]
            x_0_flat = x_0.reshape(batch_size, num_samples, latent_dim)
            x_target_flat = x_target.reshape(batch_size, num_samples, latent_dim)
        
        # Compute pairwise squared distances for cost matrix
        # x_0_flat: [batch_size, num_samples, latent_dim]
        # x_target_flat: [batch_size, num_samples, latent_dim]
        # Cost matrix C[i,j] = ||x_0[i] - x_target[j]||^2
        x_0_expanded = x_0_flat[:, :, None, :]  # [batch_size, num_samples, 1, latent_dim]
        x_target_expanded = x_target_flat[:, None, :, :]  # [batch_size, 1, num_samples, latent_dim]
        
        # Squared Euclidean distance
        cost_matrix = jnp.sum((x_0_expanded - x_target_expanded) ** 2, axis=-1)  # [batch_size, num_samples, num_samples]
        
        # Normalize cost matrix for numerical stability
        cost_matrix = cost_matrix / (jnp.max(cost_matrix, axis=(-2, -1), keepdims=True) + 1e-8)
        
        # Initialize Sinkhorn algorithm
        # K = exp(-C / epsilon) is the kernel matrix
        K = jnp.exp(-cost_matrix / epsilon)  # [batch_size, num_samples, num_samples]
        
        # Initialize scaling vectors u and v
        # Uniform marginals (equal weights for all samples)
        u = jnp.ones((batch_size, num_samples)) / num_samples  # [batch_size, num_samples]
        v = jnp.ones((batch_size, num_samples)) / num_samples  # [batch_size, num_samples]
        
        # Sinkhorn iterations
        def sinkhorn_step(carry, _):
            u, v = carry
            # Update v: v = 1 / (K^T @ u)
            v = 1.0 / (jnp.sum(K * u[:, :, None], axis=1) + 1e-10)  # [batch_size, num_samples]
            # Update u: u = 1 / (K @ v)
            u = 1.0 / (jnp.sum(K * v[:, None, :], axis=2) + 1e-10)  # [batch_size, num_samples]
            return (u, v), None
        
        # Run Sinkhorn iterations
        (u, v), _ = jax.lax.scan(sinkhorn_step, (u, v), None, length=num_iterations)
        
        # Compute transport plan P = diag(u) @ K @ diag(v)
        # P[i,j] = u[i] * K[i,j] * v[j]
        P = u[:, :, None] * K * v[:, None, :]  # [batch_size, num_samples, num_samples]
        
        # Find unique one-to-one assignment using greedy matching
        # This ensures each x_0 is assigned to exactly one x_target
        def greedy_assignment(P_batch):
            """
            Greedy assignment algorithm to find unique one-to-one matching.
            
            Args:
                P_batch: Transport plan [num_samples, num_samples]
                
            Returns:
                assignment: [num_samples] where assignment[j] = i means x_0[i] -> x_target[j]
            """
            num_samples = P_batch.shape[0]
            assignment = jnp.full((num_samples,), -1, dtype=jnp.int32)
            used_x0 = jnp.zeros(num_samples, dtype=bool)
            
            # Create a mask to track available assignments
            P_work = P_batch.copy()
            
            # Greedily assign: find max in remaining matrix, assign, mask row/column
            def assign_step(carry, _):
                assignment, used_x0, P_work = carry
                
                # Find the maximum value in the remaining matrix
                # Mask out already used rows (x_0) and columns (x_target)
                P_masked = jnp.where(
                    used_x0[:, None] | (assignment >= 0)[None, :],
                    -jnp.inf,
                    P_work
                )
                
                # Find global max
                max_idx_flat = jnp.argmax(P_masked)
                i = max_idx_flat // num_samples
                j = max_idx_flat % num_samples
                
                # Check if we found a valid assignment (not masked)
                is_valid = jnp.isfinite(P_masked[i, j]) & ~used_x0[i] & (assignment[j] < 0)
                
                # Update assignment
                assignment = jnp.where(
                    is_valid,
                    assignment.at[j].set(i),
                    assignment
                )
                
                # Mark x_0[i] as used
                used_x0 = jnp.where(
                    is_valid,
                    used_x0.at[i].set(True),
                    used_x0
                )
                
                return (assignment, used_x0, P_work), None
            
            # Run assignment steps
            (assignment, used_x0, _), _ = jax.lax.scan(
                assign_step,
                (assignment, used_x0, P_work),
                None,
                length=num_samples
            )
            
            return assignment
        
        # Apply greedy assignment to each batch
        if batch_size == 1:
            assignment = greedy_assignment(P[0])[None, :]  # [1, num_samples]
        else:
            assignment = jax.vmap(greedy_assignment)(P)  # [batch_size, num_samples]
        
        # Reorder x_0 according to assignment
        # For each batch, reorder x_0 so that x_0_reordered[j] = x_0[assignment[j]]
        batch_indices = jnp.arange(batch_size)[:, None]  # [batch_size, 1]
        x_0_reordered_flat = x_0_flat[batch_indices, assignment]  # [batch_size, num_samples, latent_dim]
        
        # Reshape back to original shape
        if len(original_shape) == 2:
            x_0_reordered = x_0_reordered_flat[0]  # [num_samples, latent_dim]
        else:
            x_0_reordered = x_0_reordered_flat.reshape(original_shape)
        
        return x_0_reordered



def create_flow_planner(
    config_dict: Dict[str, Any],
    latent_dim: int
) -> nn.Module:
    """
    Factory function to create a FlowPlanner instance.
    
    Args:
        config_dict: Configuration dictionary with flow planner settings. Supported keys:
            - top_k: Number of top clusters to sample from (default: 1)
            - sample_method: Sampling method - "mixture" (GMM) or "normal" (default: "mixture")
            - sinkhorn_refinement: Whether to enable sinkhorn refinement (default: False)
            - alpha_min, alpha_max, sigma_min, sigma_max: Flow schedule parameters
            - gmm: Nested dictionary with GMM parameters:
                - num_clusters: Number of GMM clusters (default: 512)
                - shared_variances: Whether to tie precisions across clusters (default: False)
                - prior_mu, prior_alpha, prior_beta, prior_alpha_mix, beta_mix: GMM prior parameters
            - (Backward compatibility: GMM parameters can also be at top level)
        latent_dim: Dimension of latent space (flattened dimension, e.g., 2 for (2,) or 96 for (48, 2))
        
    Returns:
        FlowPlanner instance
    """
    # Ensure latent_dim is an integer
    latent_dim = int(latent_dim)
    
    # Get configuration values with defaults
    # Support both nested (gmm.*) and flat structure for backward compatibility
    gmm_config_dict = config_dict.get("gmm", {})
    # Convert FrozenDict to dict if needed, and check if it's empty
    if hasattr(gmm_config_dict, 'unfreeze'):
        gmm_config_dict = gmm_config_dict.unfreeze()
    elif hasattr(gmm_config_dict, '__len__') and len(gmm_config_dict) == 0:
        gmm_config_dict = {}
    
    if not gmm_config_dict:
        # Backward compatibility: try flat structure
        gmm_config_dict = {
            "num_clusters": config_dict.get("num_clusters", 512),
            "shared_variances": config_dict.get("shared_variances", False),
            "prior_mu": config_dict.get("prior_mu", 0.0),
            "prior_alpha": config_dict.get("prior_alpha", 1.0),
            "prior_beta": config_dict.get("prior_beta", 1.0),
            "prior_alpha_mix": config_dict.get("prior_alpha_mix", 0.5),
            "beta_mix": config_dict.get("beta_mix", 0.1),
        }
    
    num_clusters = gmm_config_dict.get("num_clusters", 512)
    shared_variances = gmm_config_dict.get("shared_variances", False)
    prior_mu = gmm_config_dict.get("prior_mu", 0.0)
    prior_alpha = gmm_config_dict.get("prior_alpha", 1.0)
    prior_beta = gmm_config_dict.get("prior_beta", 1.0)
    prior_alpha_mix = gmm_config_dict.get("prior_alpha_mix", 0.5)
    beta_mix = gmm_config_dict.get("beta_mix", 0.1)
    
    top_k = config_dict.get("top_k", 1)
    sample_method = config_dict.get("sample_method", "mixture")
    sinkhorn_refinement = config_dict.get("sinkhorn_refinement", False)
    
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
    
    return GMMFlowPlanner(
        ndims=1,  # Number of dimensions in x_shape (latent_dim is a single dimension)
        learnable=learnable,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        gmm_config=gmm_config,
        top_k=top_k,
        sample_method=sample_method,
        sinkhorn_refinement=sinkhorn_refinement
    )
