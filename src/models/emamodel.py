# Draws inspiration from Timm's ModelEmaV2
# <https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/model_ema.py#L83>

from copy import deepcopy

import jax
import flax.linen as nn


class EmaModel(nn.Module):
    """
    Exponential Moving Average (EMA) model wrapper.

    This class implements an EMA model, which maintains a moving average of the weights
    of another model. This can be useful for model smoothing and improving generalization.

    Example usage can be Sharpness-Aware Training for Free[1].

    Attributes:
        model (nn.Module): The underlying model to apply EMA to.
        decay (float): The decay rate for the moving average. Default is 0.9999.

    References:
        [1] Du, et al. Sharpness-Aware Training for Free (2022). https://arxiv.org/abs/2205.14083
    """

    model: nn.Module
    decay: float = 0.9999

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        """
        Call the underlying model.

        This method allows the EMA model to be used as a drop-in replacement for the original model.

        Args:
            x: Input tensor.
            *args: Positional arguments to pass to the underlying model.
            **kwargs: Keyword arguments to pass to the underlying model.

        Returns:
            The output of the underlying model.
        """
        # In flax.linen, we need to create the model within the __call__ method
        # For EMA, we'll just call the model directly
        return self.model(x, *args, **kwargs)

    def initialize_ema(self, params):
        """
        Initialize the EMA parameters.

        This method creates a copy of the initial model parameters to start the EMA process.

        Args:
            params: The initial parameters of the model.
        """
        self.ema_params = jax.tree.map(lambda x: x.copy(), params)

    def update(self, params):
        """
        Update the EMA parameters.

        This method updates the EMA parameters based on the current model parameters.

        Args:
            params: The current parameters of the model.
        """
        if self.ema_params is None:
            self.initialize_ema(params)
        else:
            self.ema_params = jax.tree.map(
                lambda ema, new: self.decay * ema + (1.0 - self.decay) * new,
                self.ema_params,
                params,
            )

        # Note: In flax.linen, parameter updates are handled differently
        # This would need to be implemented using the apply function with new parameters
        # For now, we'll keep the ema_params for external use
        pass

    def set(self, params):
        """
        Set the EMA parameters directly.

        This method allows for manual setting of the EMA parameters.

        Args:
            params: The parameters to set as the new EMA parameters.
        """
        self.ema_params = jax.tree.map(lambda x: x.copy(), params)

    def get_ema_params(self):
        """
        Get the current EMA parameters.

        Returns:
            The current EMA parameters.
        """
        return self.ema_params
