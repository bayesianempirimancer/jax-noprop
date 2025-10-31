import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
from flax.core import FrozenDict
import optax
from typing import Tuple, Dict
from dataclasses import dataclass, field

from functools import partial, cached_property

# Import directly without going through src package to avoid einops dependency
from src.configs.base_config import BaseConfig
from src.utils.ode_integration import integrate_ode
from src.models.vae.encoders import create_encoder
from src.models.vae.decoders import create_decoder
from src.flow_models.crn import create_conditional_resnet


@dataclass(frozen=True)
class VAEFlowConfig(BaseConfig):
    """Configuration for VAE with flow model using separate dictionaries."""
    # BaseConfig fields
    model_name: str = "vae_flow_network"

    main: FrozenDict = field(default_factory=lambda: FrozenDict({
        "input_shape": "NA",  # Will be set based on z_dim
        "output_shape": "NA",  # Will be set based on z_dim or z_dim**2
        "latent_shape": "NA",  # Will be set based on x_dim
        "recon_loss_type": "cross_entropy", # Options: "cross_entropy", "mse", "none".  Should be consistent with decoder type
        "recon_weight": 0.1,  # Weight for reconstruction loss in total loss
        "reg_weight": 0.0,  # Weight for regularization loss in total loss
        "integration_method": "euler",  # Options: "euler", "heun", "rk4", "adaptive", "midpoint"
                                           # Only use midpoint for diffusion model.  singularities at t=0 break forward integration.
    }))

    crn: FrozenDict = field(default_factory=lambda: FrozenDict({
        "model_type": "vanilla", # Options: "vanilla", "geometric", "potential", "natural"
        "network_type": "mlp", # Options: "mlp", "bilinear", "convex"
        "hidden_dims": (64, 64, 64),
        "time_embed_dim": 64,
        "time_embed_method": "sinusoidal",
        "activation_fn": "swish",
        "use_batch_norm": False,
        "dropout_rate": 0.1,
    }))
    
    encoder: FrozenDict = field(default_factory=lambda: FrozenDict({
        "model_type": "linear", # Options: "mlp", "mlp_normal", "resnet", "resnet_normal", "identity", "linear"
        "encoder_type": "deterministic",  # Options: "deterministic", "normal", 
        "input_shape": "NA",  # Will be set from main config if not specified
        "latent_shape": "NA",
        "hidden_dims": (8,),
        "activation": "swish",
        "dropout_rate": 0.1,
    }))
    
    decoder: FrozenDict = field(default_factory=lambda: FrozenDict({
        "model_type": "identity", # Options: "mlp", "resnet", "identity"
        "decoder_type": "none", # Options: "linear", "softmax", "none"
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
            self.config.crn,
            latent_shape=self.z_shape,
            input_shape=self.config.main["input_shape"],
            output_shape=self.z_shape
        )
    
    def _broadcast_inputs(self, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        z_batch_shape = z.shape[:-self.z_ndims]
        x_batch_shape = () if x is None else x.shape[:-self.x_ndims]
        t_batch_shape = () if t is None else jnp.asarray(t).shape
        batch_shape = jnp.broadcast_shapes(z_batch_shape, x_batch_shape, t_batch_shape)

        z = jnp.broadcast_to(z, batch_shape + self.z_shape)
        x = None if x is None else jnp.broadcast_to(x, batch_shape + x.shape[-self.x_ndims:])
        t = None if t is None else jnp.broadcast_to(t, batch_shape)
        return z, x, t

    @property
    def z_shape(self) -> Tuple[int, ...]:
        """Effective z_shape from config."""
        return self.config.main["latent_shape"]
    
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
        return len(self.config.main["input_shape"])

    @property
    def y_ndims(self) -> int:
        """Number of dimensions in y_shape."""
        return len(self.config.main["output_shape"])


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
    def pred_noise(self, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """Flow model that computes dz/dt using CRN."""
        # Optimized: minimize broadcasting and avoid redundant operations
        t = jnp.asarray(t)
        z, x, t = self._broadcast_inputs(z, x, t)
                
        # Use the CRN model created in setup()
        pred_noise = self.crn_model(z, x, t, training=training)        
        return pred_noise
    
    @nn.compact
    def encoder(self, x: jnp.ndarray, training: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Encoder that maps x to latent space using single factory function.        
        Returns:
            Tuple of (mu, logvar) where:
            - For normal encoders: encoder returns (mu, logvar) directly
            - For deterministic encoders: encoder returns z, we return (z, -jnp.inf)
        """
        encoder = create_encoder(
            self.config.encoder,
            input_shape=(x.shape[-1],),  # Use flattened input dimension
            latent_shape=self.z_shape  # Use structured latent shape
        )
        encoder_output = encoder(x, training)
        if isinstance(encoder_output, tuple):
            return encoder_output              # Normal encoder returns (mu, logvar) tuple
        else:
            return encoder_output, -jnp.inf    # Deterministic encoder returns z, we return (z, -jnp.inf) for consistency

    @nn.compact
    def decoder(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """Decoder that maps latent z to output space using single factory function."""
        decoder = create_decoder(
            self.config.decoder,
            latent_shape=self.z_shape,  # Use structured latent shape
            output_shape=self.config.main["output_shape"]  # Use full output_shape
        )
        return decoder(x, training)

            
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray, key: jr.PRNGKey, training: bool = True) -> jnp.ndarray:
        # For initialization, we need to call the nn compact methods to initialize parameters

        batch_shape = y.shape[:-self.y_ndims]
                
        # Call pred_noise to initialize its parameters (need dummy z and t)
        dummy_z = jnp.zeros(batch_shape + self.z_shape)
        dummy_t = jnp.zeros(batch_shape)
        
        # Call pred_noise to initialize the CRN model parameters
        flow_output = self.pred_noise(dummy_z, x, dummy_t, training)
        
        # Call encoder and decoder to initialize their parameters
        # These @nn.compact methods need to be called during initialization to create parameters
        encoder_output = self.encoder(y, training)
        decoder_output = self.decoder(dummy_z, training)
        
        # For initialization, we just return a dummy output
        # The actual forward pass logic is handled by the individual methods
        return jnp.zeros(batch_shape + self.z_shape)

    @partial(jax.jit, static_argnums=(0, 5))  # self, num_steps, integration_method, output_type, and training are static arguments
    def loss(self, params: dict, x: jnp.ndarray, y: jnp.ndarray, key: jr.PRNGKey, training: bool = True) -> Tuple[jnp.ndarray, dict]:
        """
        Compute the diffusion loss.
        
        For diffusion, the loss is MSE between predicted noise and actual noise:
        L_diff = E[||model_output - noise||²/beta], where beta is 1/t for the linear noise schedule.
        """
        key, encoder_key, t_key, noise_key = jr.split(key, 4)
        key, dropout_key1, dropout_key2  = jr.split(key, 3)
        batch_shape = y.shape[:-self.y_ndims]
        # Encode Target (noisy latent)
        mu_z_target, logvar_z_target = self.apply(params, y, method='encoder', training=training, rngs={'dropout': encoder_key})
        z_target = mu_z_target + jnp.exp(0.5 * logvar_z_target) * jr.normal(key, mu_z_target.shape)
                
        # Sample noise and time
        t = jr.uniform(t_key, batch_shape + self.z_ndims*(1,), minval=0.0, maxval=1.0)
        t_squeezed = t.squeeze(tuple(range(-self.z_ndims, 0)))
        noise = jr.normal(noise_key, z_target.shape)

        # Compute noise schedule parameters
        alpha, gamma_prime = self.apply(params, t, method='get_alpha_bar_gamma_prime_t')
        sqrt_alpha = jnp.sqrt(alpha)
        sqrt_1_minus_alpha = jnp.sqrt(1-alpha)

        use_snr_weight = self.config.main.get("use_snr_weight", True)
        if use_snr_weight:
            snr_weight = (1.0/(1-alpha)/gamma_prime).squeeze(tuple(range(-self.z_ndims, 0)))  # alpha/alpha_prime
        else:
            snr_weight = 1.0

        # Compute noisy latent state at time t and predicted noise, and predicted output
        z_t =  sqrt_alpha* z_target +  sqrt_1_minus_alpha * noise # noisy latent
        predicted_noise = self.apply(params, z_t, x, t, method='pred_noise', training=training, rngs={'dropout': dropout_key1})
        z_target_est = (z_t - predicted_noise * sqrt_1_minus_alpha)/(sqrt_alpha + 1e-4)
        y_pred = self.apply(params, z_target_est, method='decoder', training=training, rngs={'dropout': dropout_key2})
        
        # Compute Losses
        squared_error = jnp.mean((predicted_noise - noise) ** 2, axis=tuple(range(-self.z_ndims, 0)))  # has batch_shape
        snr_weighted_loss = jnp.mean(squared_error*snr_weight)        

        reg_loss = jnp.mean(self.dz_dt_lazy(z_t, t, predicted_noise, alpha, gamma_prime)**2, axis=tuple(range(-self.z_ndims, 0)))
        reg_loss = jnp.mean(snr_weight * reg_loss)

        recon_loss_type = self.config.main.get("recon_loss_type", "none")
        if recon_loss_type == "cross_entropy":   recon_loss = -jnp.mean(y * jnp.log(y_pred + 1e-8), axis=tuple(range(-self.y_ndims, 0)))
        else:                                    recon_loss =  jnp.mean((y - y_pred)**2, axis=tuple(range(-self.y_ndims, 0)))
        recon_loss = jnp.mean(snr_weight * recon_loss)

        # Reconstruction loss
        reg_weight = self.config.main.get("reg_weight", 0.0)
        recon_weight = self.config.main.get("recon_weight", 0.0)
        if recon_loss_type == "none":  recon_weight = 0.0

        total_loss = snr_weighted_loss + recon_weight * recon_loss + reg_weight * reg_loss
        
        return total_loss, {
            'flow_loss': snr_weighted_loss,  # Add separate diffusion_loss metric
            'recon_loss': recon_loss, 
            'reg_loss': reg_loss,
            'total_loss': total_loss
        }

    
    def dz_dt(self, params: dict, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Compute dz/dt (diffusion vector field) for given z, x, t.
        For diffusion: dz/dt = 0.5*z - predicted_noise
        """
        predicted_noise = self.apply(params, z, x, t, method='pred_noise', training=training)

        alpha_prime = 1.0
        alpha = t
        """
        Usual Equation:       dz_dt = 0.5*alpha_prime/alpha * z - predicted_noise*alpha_prime/alpha/jnp.sqrt(1-alpha)

        Rescaled Equation:   alpha = sigmoid(gamma(t)),  alpha_prime = gamma_prime(t)*alpha*(1-alpha)
                                
                      -->    dz_ds = ( z*(1-alpha)/2 - predicted_noise*sqrt(1-alpha) ) * gamma_prime
        """
        return alpha_prime/alpha*(0.5*z  - predicted_noise/jnp.sqrt(1-alpha))


    @partial(jax.jit, static_argnums=(0, 3, 4, 5))  # self, num_steps, integration_method, output_type, and training are static arguments
    def predict(self, params: dict, x: jnp.ndarray, num_steps: int = 20, integration_method: str = "midpoint", output_type: str = "end_point", prng_key: jr.PRNGKey = None) -> jnp.ndarray:
        """
        Generate predictions using ODE solver integration
        """
        # Disable gradient tracking through parameters for inference
        params_no_grad = jax.lax.stop_gradient(params)
        batch_shape = x.shape[:-self.x_ndims]

        # Generate flattened initial latent state z_0
        if prng_key is not None:
            z_0 = jr.normal(prng_key, batch_shape + (self.z_dim,))
        else:
            z_0 = jnp.zeros(batch_shape + (self.z_dim,))  # ode expects vectorized z
        
        # Define the diffusion vector field: dz/dt = 0.5*z - predicted_noise
        def vector_field(params, z, x, t):
            z = self._unflatten_z(z)
            dz_dt = self.dz_dt(params, z, x, t)
            return self._flatten_z(dz_dt)
        
        # Integrate the ODE from t=0 to t=1 (reverse diffusion process)
        z = integrate_ode(
            vector_field=vector_field,
            params=params_no_grad,
            z0=z_0,
            x=x,
            time_span=(0.0, 1.0),
            num_steps=num_steps,
            method=integration_method,
            output_type=output_type
        )
        z = self._unflatten_z(z)
        return self.apply(params, z, method='decoder', training=False)

    @partial(jax.jit, static_argnums=(0, 3, 4, 5, 6))  # self, num_steps, integration_method, output_type are static arguments
    def sample(self, params: dict, prng_key: jr.PRNGKey, batch_shape: Tuple[int, ...], num_steps: int = 20, integration_method: str = "midpoint", output_type: str = "end_point") -> jnp.ndarray:
        """
        Generate samples using diffusion sampling without conditional input.
        Returns:
            If output_type="end_point": Final samples [batch_shape + y_shape]
            If output_type="trajectory": Full trajectory [num_steps, batch_shape + y_shape]
        """        
        params_no_grad = jax.lax.stop_gradient(params)
        z_0 = jr.normal(prng_key, batch_shape + (self.z_dim,))
        # Define the vector field for ODE integration using flow_model method with x=None
        def vector_field(params, z, x, t):
            z = self._unflatten_z(z)
            dz_dt = self.dz_dt(params, z, None, t, training=False)  # Use x=None
            return self._flatten_z(dz_dt)
        
        # Integrate the ODE from t=0 to t=1 (reverse diffusion process)
        z = integrate_ode(
            vector_field=vector_field,
            params=params_no_grad,
            z0=z_0,
            x=None,  # No conditional input
            time_span=(0.0, 1.0),
            num_steps=num_steps,
            method=integration_method,
            output_type=output_type
        )
        
        z = self._unflatten_z(z)
        return self.apply(params, z, method='decoder', training=False)


    @partial(jax.jit, static_argnums=(0, 5, 7))  # self, optimizer, and training are static arguments
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
    
    def train_step_without_dropout(self, params: dict, x: jnp.ndarray, y: jnp.ndarray, opt_state: dict, optimizer, key: jr.PRNGKey, training: bool = False) -> Tuple[dict, dict, jnp.ndarray, dict]:
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