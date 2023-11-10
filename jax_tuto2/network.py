import flax.linen as nn
from jax import random
import numpy as np
from typing import Sequence
from activation_fct import Sigmoid


# To keep the results close to the PyTorch tutorial, we use the same init
# function as PyTorch
# which is uniform(-1/sqrt(in_features), 1/sqrt(in_features))
# - similar to He et al./kaiming
# The default for Flax is lecun_normal (i.e., half the variance of He)
# and zeros for bias.
def init_func(x):
    return (
        lambda rng, shape, dtype: random.uniform(
            rng,
            shape=shape,
            minval=-1 / np.sqrt(x.shape[1]),
            maxval=1 / np.sqrt(x.shape[1]),
            dtype=dtype,
        )
    )


# Network
class BaseNetwork(nn.Module):
    act_fn: nn.Module
    num_classes: int = 10
    hidden_sizes: Sequence = (512, 256, 256, 128)

    @nn.compact
    def __call__(self, x, return_activations=False):
        x = x.reshape(x.shape[0], -1)  # Reshape images to a flat vector
        # We collect all activations throughout the network for later visualizations
        # Remember that in jitted functions, unused tensors will anyways be removed.
        activations = []
        for hd in self.hidden_sizes:
            x = nn.Dense(hd, kernel_init=init_func(x), bias_init=init_func(x))(x)
            activations.append(x)
            x = self.act_fn(x)
            activations.append(x)
        x = nn.Dense(
            self.num_classes, kernel_init=init_func(x), bias_init=init_func(x)
        )(x)
        return x if not return_activations else (x, activations)


if __name__ == "__main__":
    pass
    # model = BaseNetwork(Sigmoid)
