import torch
import torch.nn as nn
from tqdm import tqdm

from activation_function import ActivationFunction, ReLU
from dataset import train_loader, set_seed, CHECKPOINT_PATH
from network import BaseNetwork, load_model

device = (
    torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
)


def measure_number_dead_neurons(net):
    # For each neuron, we create a boolean variable initially set to 1. If it has an activation unequals 0 at any time,
    # we set this variable to 0. After running through the whole training set, only dead neurons will have a 1.
    neurons_dead = [
        torch.ones(layer.weight.shape[0], device=device, dtype=torch.bool)
        for layer in net.layers[:-1]
        if isinstance(layer, nn.Linear)
    ]  # Same shapes as hidden size in BaseNetwork

    net.eval()
    with torch.no_grad():
        for imgs, _ in tqdm(
            train_loader, leave=False
        ):  # Run through whole training set
            layer_index = 0
            imgs = imgs.to(device)
            imgs = imgs.view(imgs.size(0), -1)
            for layer in net.layers[:-1]:
                imgs = layer(imgs)
                if isinstance(layer, ActivationFunction):
                    # Are all activations == 0 in the batch, and we did not record the opposite in the last batches?
                    neurons_dead[layer_index] = torch.logical_and(
                        neurons_dead[layer_index], (imgs == 0).all(dim=0)
                    )
                    layer_index += 1
    number_neurons_dead = [t.sum().item() for t in neurons_dead]
    print("Number of dead neurons:", number_neurons_dead)
    print(
        "In percentage:",
        ", ".join(
            [
                f"{(100.0 * num_dead / tens.shape[0]):4.2f}%"
                for tens, num_dead in zip(neurons_dead, number_neurons_dead)
            ]
        ),
    )


if __name__ == "__main__":
    set_seed(42)

    # net_relu = BaseNetwork(act_fn=ReLU()).to(device)
    # measure_number_dead_neurons(net_relu)

    ####################################################

    # net_relu = load_model(model_path=CHECKPOINT_PATH, model_name="FashionMNIST_relu").to(device)
    # measure_number_dead_neurons(net_relu)

    ####################################################

    net_relu = BaseNetwork(
        act_fn=ReLU(), hidden_sizes=[256, 256, 256, 256, 256, 128, 128, 128, 128, 128]
    ).to(device)
    measure_number_dead_neurons(net_relu)
