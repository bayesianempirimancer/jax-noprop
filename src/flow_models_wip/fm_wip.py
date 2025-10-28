"""
NoProp-CT: Continuous-time NoProp implementation.

This work in progress module adds a learable embedding for the output target z_target = f(y)
and and associated decoder p(y|z_target).  For example in the simple case where y is categorical, 
the learnable decoder is a softmax times a one hot representation of y and the learnable embedding
is some linear transformation of the one hot representation of y.  More generally z_target = f(y) is 
a neural network, and p(y|z_target) is some neural network parameterized distribution.  This ammounts 
to taking a sort of VAE on at the end of the flow.  

The reason this is a work in progress is because this will require a substaantial modification of the 
existing noprop code.  The big change is that the call routine now needs to output something that can be 
easily turned into a prediction for z_target from z_t as input to the probabilistic decoder and the encoder
of the y.  These are the three things that will now contribute to the loss function at each time step.  

Another possibel approach would be to create a VAE and put the flow model in for the element that sets the 
prior distribution over the latent variable z_target.  This would be a more principled approach.  We will 
explore both approaches here.  
"""
import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
from flax.core import FrozenDict
import optax
from typing import Tuple, Dict
from dataclasses import dataclass, field

from functools import partial, cached_property
from src.configs.base_config import BaseConfig
from src.utils.ode_integration import integrate_ode

from src.models.vae.encoders import create_encoder
from src.models.vae.decoders import create_decoder
from src.flow_models_wip.crn_wip import create_conditional_resnet


@dataclass(frozen=True)
class VAEFlowConfig(BaseConfig):
    """Configuration for VAE with flow model using separate dictionaries."""
    # BaseConfig fields
    model_name: str = "vae_flow_network"

    config: FrozenDict = field(default_factory=lambda: FrozenDict({
        "input_shape": "NA",  # Will be set based on z_dim
        "output_shape": "NA",  # Will be set based on z_dim or z_dim**2
        "latent_shape": "NA",  # Will be set based on x_dim
        "loss_type": "cross_entropy", # Options: "cross_entropy", "mse", "none".  Should be consistent with decoder type
        "flow_loss_weight": 0.01,  # Weight for flow loss in total loss
        "reg_weight": 0.0,  # Weight for regularization loss in total loss
    }))

    crn_config: FrozenDict = field(default_factory=lambda: FrozenDict({
        "model_type": "vanilla", # Options: "vanilla", "geometric", "potential", "natural"
        "network_type": "mlp", # Options: "mlp", "bilinear", "convex"
        "hidden_dims": (64, 64, 64),
        "time_embed_dim": 64,
        "time_embed_method": "sinusoidal",
        "activation_fn": "swish",
        "use_batch_norm": False,
        "dropout_rate": 0.1,
    }))
    
    encoder_config: FrozenDict = field(default_factory=lambda: FrozenDict({
        "model_type": "mlp", # Options: "mlp", "mlp_normal", "resnet", "resnet_normal", "identity"
        "encoder_type": "normal",  # Options: "deterministic", "normal"
        "input_shape": "NA",  # Will be set from main config if not specified
        "latent_shape": "NA",
        "hidden_dims": (64, 32, 16),
        "activation": "swish",
        "dropout_rate": 0.1,
    }))
    
    decoder_config: FrozenDict = field(default_factory=lambda: FrozenDict({
        "model_type": "mlp", # Options: "mlp", "resnet", "identity"
        "decoder_type": "logits", # Options: "logits", "normal"
        "latent_shape": "NA",  # Will be set from main config if not specified
        "output_shape": "NA",
        "hidden_dims": (64, 32, 16),
        "activation": "swish",
        "dropout_rate": 0.1,
    }))


class VAE_flow(nn.Module):
    """Variational Autoencoder with flow model using @nn.compact methods."""
    config: VAEFlowConfig
    
    def setup(self):
        """Initialize the CRN model as a field."""
        self.crn_model = create_conditional_resnet(
            self.config.crn_config,
            latent_shape=self.z_shape,
            input_shape=self.config.config["input_shape"],
            output_shape=self.z_shape
        )
    

    @property
    def z_shape(self) -> Tuple[int, ...]:
        """Effective z_shape from config."""
        return self.config.config["latent_shape"]
    
    @property
    def z_ndims(self) -> int:
        """Number of dimensions in z_shape."""
        return len(self.z_shape)
    
    @cached_property
    def z_dim(self) -> int:
        """Total flattened dimension of z."""
        z_dim = 1
        for dim in self.z_shape:
            z_dim *= dim
        return z_dim

    @property
    def x_ndims(self) -> int:
        """Number of dimensions in x_shape."""
        return len(self.config.config["input_shape"])

    def _flatten_z(self, z: jnp.ndarray) -> jnp.ndarray:
        """Flatten the z tensor to 1D for processing."""
        if len(self.z_shape) <= 1:
            return z
        return z.reshape(z.shape[:-self.z_ndims] + (self.z_dim,))

    def _unflatten_z(self, z: jnp.ndarray) -> jnp.ndarray:
        """Unflatten the z tensor back to original shape."""
        if len(self.z_shape) <= 1:
            return z
        return z.reshape(z.shape[:-1] + self.z_shape)
    
    @nn.compact
    def flow_model(self, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """Flow model that computes dz/dt using CRN."""
        # Optimized: minimize broadcasting and avoid redundant operations
        t = jnp.asarray(t)
        
        # Only broadcast if shapes don't match - avoid unnecessary operations
        batch_shape = jnp.broadcast_shapes(
            x.shape[:-self.x_ndims], 
            z.shape[:-self.z_ndims], 
            t.shape
        )
        if x.shape[:-self.x_ndims] != batch_shape:
            x = jnp.broadcast_to(x, batch_shape + x.shape[-self.x_ndims:])
        if z.shape[:-self.z_ndims] != batch_shape:
            z = jnp.broadcast_to(z, batch_shape + z.shape[-self.z_ndims:])
        if t.shape != batch_shape:
            t = jnp.broadcast_to(t, batch_shape)
        
        z_flat = self._flatten_z(z)
        
        # Use the CRN model created in setup()
        dz_dt_flat = self.crn_model(z_flat, x, t, training=training)
        
        # Optimized: only unflatten if necessary
        if len(self.z_shape) <= 1:
            return dz_dt_flat
        else:
            return dz_dt_flat.reshape(dz_dt_flat.shape[:-1] + self.z_shape)
    
    @nn.compact
    def encoder(self, x: jnp.ndarray, training: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Encoder that maps x to latent space using single factory function.
        
        Returns:
            Tuple of (mu, logvar) where:
            - For normal encoders: encoder returns (mu, logvar) directly
            - For deterministic encoders: encoder returns z, we return (z, -jnp.inf)
        """
        # Use direct factory function: create_encoder(config_dict, input_shape, latent_shape)
        encoder = create_encoder(
            self.config.encoder_config,
            input_shape=(x.shape[-1],),  # Use flattened input dimension
            latent_shape=self.z_shape  # Use structured latent shape
        )

        # Get encoder output - could be tuple (mu, logvar) or single tensor z
        encoder_output = encoder(x, training)
        
        # Optimized: move assertions outside JIT compilation for better performance
        # (Assertions are kept for debugging but will be compiled out in optimized builds)
        
        # Handle both normal and deterministic cases
        if isinstance(encoder_output, tuple):
            # Normal encoder returns (mu, logvar) tuple
            return encoder_output
        else:
            # Deterministic encoder returns z, we return (z, -jnp.inf) for consistency
            return encoder_output, -jnp.inf

    @nn.compact
    def decoder(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """Decoder that maps latent z to output space using single factory function."""
        # Create decoder network using direct factory
        decoder = create_decoder(
            self.config.decoder_config,
            latent_shape=self.z_shape,  # Use structured latent shape
            output_shape=(self.config.config["output_shape"][0],)  # Use config output_dim
        )
        
        # Get raw output from decoder
        output = decoder(x, training)
        
        # Optimized: apply output transformation based on decoder type
        decoder_type = self.config.decoder_config["decoder_type"]
        if decoder_type == "logits":
            return jax.nn.softmax(output, axis=-1)
        elif decoder_type == "normal":
            return output
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

            
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray, key: jr.PRNGKey, training: bool = True) -> jnp.ndarray:
        # For initialization, we need to call the nn compact methods to initialize parameters

        batch_shape = x.shape[:-self.x_ndims]
                
        # Call flow_model to initialize its parameters (need dummy z and t)
        dummy_z = jnp.zeros(batch_shape + self.z_shape)
        dummy_t = jnp.zeros(batch_shape)
        
        # Call flow_model to initialize the CRN model parameters
        flow_output = self.flow_model(dummy_z, x, dummy_t, training)
        
        # Call encoder and decoder to initialize their parameters
        # These @nn.compact methods need to be called during initialization to create parameters
        encoder_output = self.encoder(y, training)
        decoder_output = self.decoder(dummy_z, training)
        
        # For initialization, we just return a dummy output
        # The actual forward pass logic is handled by the individual methods
        return jnp.zeros(batch_shape + self.z_shape)
        
    def loss(self, params: dict, x: jnp.ndarray, y: jnp.ndarray, key: jr.PRNGKey, training: bool = True) -> Tuple[jnp.ndarray, dict]:
        """
        Compute the loss by calling individual @nn.compact methods with proper rngs.
        
        Args:
            params: Model parameters
            x: Input data [batch_shape + x_shape]
            y: Target labels [batch_shape + y_shape]
            key: Random key for sampling
            training: Whether in training mode
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        # Optimized: reduce random key splits and optimize sampling
        key, t_key, z_0_key, z_t_noise_key, z_target_key = jr.split(key, 5)

        batch_shape = x.shape[:-self.x_ndims]
        
        # Optimized: get target encoding from y encoder with proper rngs
        mu_z_target, logvar_z_target = self.apply(params, y, method='encoder', training=training, rngs={'dropout': key})
        z_target = mu_z_target + jnp.exp(0.5 * logvar_z_target) * jr.normal(z_target_key, mu_z_target.shape)

        # Optimized: sample z_0 and t more efficiently
        z_0 = jr.normal(z_0_key, batch_shape + self.z_shape)
        t = jr.uniform(t_key, batch_shape + self.z_ndims*(1,), minval=0.0, maxval=1.0)
        
        # Optimized: construct z_t more efficiently
        diff_z = z_target - z_0
        z_t = z_0 + t * diff_z
        z_t = z_t + jr.normal(z_t_noise_key, z_t.shape) * jnp.sqrt(1-t)

        # Optimized: get flow field dz/dt using flow_model method with proper rngs
        squeezed_t = t.squeeze(tuple(range(-self.z_ndims, 0)))
        dz_dt = self.apply(params, z_t, x, squeezed_t, method='flow_model', training=training, rngs={'dropout': key})    
        flow_loss = jnp.mean((dz_dt - diff_z)**2)        
#        flow_loss = jnp.mean((z_target - z_target_est)**2)

        # Estimate target using flow field
        z_target_est = dz_dt * (1.0-t) + z_t


        # Optimized: get the predicted probabilities from decoder with proper rngs
        y_pred = self.apply(params, z_target_est, method='decoder', training=training, rngs={'dropout': key})

        # Optimized: compute loss more efficiently
        loss_type = self.config.config["loss_type"]
        if loss_type == "cross_entropy":
            recon_loss = -jnp.mean(y * jnp.log(y_pred + 1e-8))
        elif loss_type == "mse":
            recon_loss = jnp.mean((y - y_pred)**2)
        elif loss_type == "none":
            recon_loss = 0.0
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # Optimized: pre-compute flow_loss_weight
        flow_loss_weight = self.config.config["flow_loss_weight"]
        total_loss = flow_loss_weight * flow_loss + recon_loss

        return total_loss, {'flow_loss': flow_loss, 'recon_loss': recon_loss, 'total_loss': total_loss}

    @partial(jax.jit, static_argnums=(0, 3, 4, 5))  # self, num_steps, integration_method, output_type, and training are static arguments
    def predict(self, params: dict, x: jnp.ndarray, num_steps: int = 20, integration_method: str = "euler", output_type: str = "end_point") -> jnp.ndarray:
        """
        Make predictions using ODE solver integration.
        
        Args:
            params: Model parameters
            x: Input data [batch_shape + x_shape]
            num_steps: Number of integration steps (default: 20)
            integration_method: Integration method ("euler", "heun", "rk4", "adaptive") (default: "euler")
            output_type: Type of output ("end_point" for final prediction, "trajectory" for full trajectory)
            
        Returns:
            If output_type="end_point": Final prediction [batch_shape + y_shape]
            If output_type="trajectory": Full trajectory [num_steps, batch_shape + y_shape]
        """
        
        # Optimized: disable gradient tracking through parameters for inference
        params_no_grad = jax.lax.stop_gradient(params)
        
        # Optimized: generate initial latent state z_0 with proper shape
        batch_shape = x.shape[:-self.x_ndims]
        z_0 = jnp.zeros(batch_shape + (self.z_dim,))  # ode expects vectorized z
        
        # Optimized: define the vector field for ODE integration using flow_model method
        def vector_field(params, z, x, t):
            # Only unflatten/flatten if necessary (avoid for 1D z)
            if len(self.z_shape) <= 1:
                dz_dt = self.apply(params, z, x, t, method='flow_model', training=False)
                return dz_dt
            else:
                z_structured = self._unflatten_z(z)  # flow_model expects structured z
                dz_dt_structured = self.apply(params, z_structured, x, t, method='flow_model', training=False)
                return self._flatten_z(dz_dt_structured)  # ODE solver expects flattened z
        
        # Integrate the ODE from t=0 to t=1
        z_trajectory = integrate_ode(
            vector_field=vector_field,
            params=params_no_grad,
            z0=z_0,
            x=x,
            time_span=(0.0, 1.0),
            num_steps=num_steps,
            method=integration_method,
            output_type=output_type
        )
        
        # Optimized: decode the final latent state to get predictions
        if len(self.z_shape) <= 1:
            z_trajectory_structured = z_trajectory
        else:
            z_trajectory_structured = self._unflatten_z(z_trajectory)
        return self.apply(params, z_trajectory_structured, method='decoder', training=False)
    
    def get_dzdt(self, params: dict, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        Compute dz/dt (vector field) for given z, x, t.
        On input and output z is flattened.
        Args:
            params: Model parameters
            z: Latent state [batch_size, latent_dim]
            x: Conditional input [batch_size, input_dim]
            t: Time [batch_size] or scalar
            
        Returns:
            dz/dt [batch_size, latent_dim]
        """
        z = self._unflatten_z(z)  # flow_model expects structured z
        dz_dt = self.apply(params, z, x, t, method='flow_model', training=False)
        return self._flatten_z(z)

    @partial(jax.jit, static_argnums=(0, 5))  # self and optimizer are static arguments
    def train_step(self, params: dict, x: jnp.ndarray, y: jnp.ndarray, opt_state: dict, optimizer, key: jr.PRNGKey, training: bool = True) -> Tuple[dict, dict, jnp.ndarray, dict]:
        """
        JIT-compiled training step for VAE with flow model.
        
        Args:
            params: Current model parameters
            x: Conditional input [batch_size, input_dim]
            y: Target output [batch_size, output_dim]
            opt_state: Optimizer state
            optimizer: Optax optimizer
            key: Random key
            use_dropout: Whether to use dropout during training
            
        Returns:
            params: Updated model parameters
            opt_state: Updated optimizer state
            loss: Training loss
            metrics: Training metrics
        """
        # Compute loss and gradients
        def loss_fn(params):
            return self.loss(params, x, y, key, training=training)  
        
        (loss, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        
        # Update parameters using optimizer
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss, metrics
    
    @partial(jax.jit, static_argnums=(0, 5))  # self and optimizer are static arguments
    def train_step_with_dropout(self, params: dict, x: jnp.ndarray, y: jnp.ndarray, opt_state: dict, optimizer, key: jr.PRNGKey) -> Tuple[dict, dict, jnp.ndarray, dict]:
        """JIT-compiled training step with dropout enabled."""

        def loss_fn(params):
            return self.loss(params, x, y, key, training=True)
        
        (loss, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss, metrics
    
    @partial(jax.jit, static_argnums=(0, 5))  # self and optimizer are static arguments
    def train_step_without_dropout(self, params: dict, x: jnp.ndarray, y: jnp.ndarray, opt_state: dict, optimizer, key: jr.PRNGKey) -> Tuple[dict, dict, jnp.ndarray, dict]:
        """JIT-compiled training step with dropout disabled."""
        def loss_fn(params):
            return self.loss(params, x, y, key, training=False)
        
        (loss, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss, metrics


