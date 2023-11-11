import math
import flax.linen as nn

from jax import random
import jax.numpy as jnp
import numpy as np
import torch.utils.data as data

from dataset import numpy_collate, train_set
from network import BaseNetwork, act_fn_by_name
from visualization import (
    visualize_activations,
    visualize_gradients,
    visualize_weight_distribution,
)


def init_simple_model(kernel_init, exmp_imgs, act_fn=act_fn_by_name["identity"]):
    model = BaseNetwork(act_fn=act_fn, kernel_init=kernel_init)
    params = model.init(random.PRNGKey(42), exmp_imgs)
    return model, params


# An initialization function in JAX takes as input a PRNG key,
# the shape of the parameter to create, and the data type
# (usually jnp.float32). We create this function based on the
# input parameter 'c' here, indicating the constant value
def get_const_init_func(c=0.0):
    return lambda key, shape, dtype: c * jnp.ones(shape, dtype=dtype)


def get_var_init_func(std=0.01):
    return lambda key, shape, dtype: std * random.normal(key, shape, dtype=dtype)


equal_var_init = (
    lambda key, shape, dtype: 1.0
    / np.sqrt(shape[0])
    * random.normal(key, shape, dtype=dtype)
)


def xavier_init(key, shape, dtype):
    bound = math.sqrt(6) / math.sqrt(shape[0] + shape[1])
    return random.uniform(key, shape, dtype, minval=-bound, maxval=bound)


def kaiming_init(key, shape, dtype):
    # The first layer does not have ReLU applied on its input
    # Note that this detection only works if we do not use 784
    # feature size anywhere - better to explicitly handle
    # layer numbers
    small_loader = data.DataLoader(
        train_set, batch_size=1024, shuffle=False, collate_fn=numpy_collate
    )
    exmp_imgs, _ = next(iter(small_loader))


    num_input_feats = np.prod(exmp_imgs.shape[1:])
    if shape[0] == num_input_feats:
        std = 1 / np.sqrt(shape[0])
    else:
        std = np.sqrt(2 / shape[0])
    return std * random.normal(key, shape, dtype)


if __name__ == "__main__":
    small_loader = data.DataLoader(
        train_set, batch_size=1024, shuffle=False, collate_fn=numpy_collate
    )
    exmp_imgs, exmp_labels = next(iter(small_loader))

    ############################################################################
    # model, params = init_simple_model(get_const_init_func(c=0.005))
    # visualize_gradients(model, params, exmp_imgs, exmp_labels)
    # visualize_activations(model, params, exmp_imgs, print_variance=True)

    ############################################################################
    # model, params = init_simple_model(get_var_init_func(std=0.01))
    # visualize_activations(model, params, exmp_imgs, print_variance=True)

    ############################################################################
    # model, params = init_simple_model(get_var_init_func(std=0.1))
    # visualize_activations(model, params, exmp_imgs, print_variance=True)

    ############################################################################
    # model, params = init_simple_model(equal_var_init)
    # visualize_weight_distribution(params)
    # visualize_activations(model, params, exmp_imgs, print_variance=True)

    ############################################################################

    # model, params = init_simple_model(xavier_init)
    # visualize_gradients(model, params, exmp_imgs, exmp_labels, print_variance=True)
    # visualize_activations(model, params, exmp_imgs, print_variance=True)

    ############################################################################
    # model, params = init_simple_model(xavier_init, act_fn=nn.tanh)
    # visualize_gradients(model, params, exmp_imgs, exmp_labels, print_variance=True)
    # visualize_activations(model, params, exmp_imgs, print_variance=True)

    ############################################################################
    num_input_feats = np.prod(exmp_imgs.shape[1:])
    model, params = init_simple_model(kaiming_init, act_fn=nn.relu)
    visualize_gradients(model, params, exmp_imgs, exmp_labels, print_variance=True)
    visualize_activations(model, params, exmp_imgs, print_variance=True)
