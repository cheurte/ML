import jax
import torch.utils.data as data
from tqdm import tqdm
import jax.numpy as jnp
from activation_fct import ReLU
from network import BaseNetwork
from jax import random
from dataset import load_dataset, numpy_collate, CHECKPOINT_PATH
from utils import load_model


def measure_number_dead_neurons(net, params, train_loader):
    # For each neuron, we create a boolean variable initially set to 1.
    # If it has an activation unequals 0 at any time,
    # we set this variable to 0.
    #  After running through the whole training set, only dead neurons will have a 1.
    neurons_dead = [
        jnp.ones(hd, dtype=jnp.dtype("bool")) for hd in net.hidden_sizes
    ]  # Same shapes as hidden size in BaseNetwork

    get_activations = jax.jit(
        lambda inp: net.apply(params, inp, return_activations=True)[1]
    )
    for imgs, _ in tqdm(train_loader, leave=False):  # Run through whole training set
        activations = get_activations(imgs)
        for layer_index, activ in enumerate(activations[1::2]):
            # Are all activations == 0 in the batch, and we did not record
            # the opposite in the last batches?
            neurons_dead[layer_index] = jnp.logical_and(
                neurons_dead[layer_index], (activ == 0).all(axis=0)
            )
    number_neurons_dead = [t.sum().item() for t in neurons_dead]
    print("Number of dead neurons:", number_neurons_dead)
    print(
        "In percentage:",
        ", ".join(
            [
                f"{num_dead / tens.shape[0]:4.2%}"
                for tens, num_dead in zip(neurons_dead, number_neurons_dead)
            ]
        ),
    )


if __name__ == "__main__":
    (
        train_dataset,
        train_set,
        val_set,
        test_set,
        train_loader,
        val_loader,
        test_loader,
    ) = load_dataset()
    small_loader = data.DataLoader(
        train_set, batch_size=256, shuffle=False, collate_fn=numpy_collate
    )
    exmp_batch = next(iter(small_loader))

    net_relu = BaseNetwork(act_fn=ReLU())
    params = net_relu.init(random.PRNGKey(42), exmp_batch[0])
    measure_number_dead_neurons(net_relu, params, train_loader)

    state, net_relu = load_model(
        model_path=CHECKPOINT_PATH, model_name="FashionMNIST_relu"
    )
    measure_number_dead_neurons(net_relu, state.params, train_loader)

    net_relu = BaseNetwork(
        act_fn=ReLU(), hidden_sizes=[256, 256, 256, 256, 256, 128, 128, 128, 128, 128]
    )
    params = net_relu.init(random.PRNGKey(42), exmp_batch[0])
    measure_number_dead_neurons(net_relu, params, train_loader)
