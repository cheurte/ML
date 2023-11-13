import warnings

import matplotlib.pyplot as plt
import seaborn as sns
import math
import torch
import torch.nn.functional as F
import torch.utils.data as data

from dataset import train_set, set_seed, CHECKPOINT_PATH
from activation_function import act_fn_by_name
from network import BaseNetwork, load_model
warnings.filterwarnings("ignore")

device = (
    torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
)


def visualize_gradients(net: BaseNetwork, color="C0"):
    """
    Inputs:
        net - Object of class BaseNetwork
        color - Color in which we want to visualize the histogram (for easier separation of activation functions)
    """
    net.eval()
    small_loader = data.DataLoader(train_set, batch_size=256, shuffle=False)
    imgs, labels = next(iter(small_loader))
    imgs, labels = imgs.to(device), labels.to(device)

    # Pass one batch through the network, and calculate the gradients for the weights
    net.zero_grad()
    preds = net(imgs)
    loss = F.cross_entropy(preds, labels)
    loss.backward()
    # We limit our visualization to the weight parameters and exclude the bias to reduce the number of plots

    grads = {
        name: params.grad.data.view(-1).cpu().clone().numpy()
        for name, params in net.named_parameters()
        if "weight" in name
    }
    net.zero_grad()

    ## Plotting
    columns = len(grads)
    fig, ax = plt.subplots(1, columns, figsize=(columns * 3.5, 2.5))
    fig_index = 0
    for key in grads:
        key_ax = ax[fig_index % columns]
        sns.histplot(data=grads[key], bins="auto", ax=key_ax, color=color, kde=True)
        key_ax.set_title(str(key))
        key_ax.set_xlabel("Grad magnitude")
        fig_index += 1
    fig.suptitle(
        f"Gradient magnitude distribution for activation function {net.config['act_fn']['name']}",
        fontsize=14,
        y=1.05,
    )
    fig.subplots_adjust(wspace=0.45)
    plt.show()
    plt.close()

def visualize_activations(net, color="C0"):
    activations = {}

    net.eval()
    small_loader = data.DataLoader(train_set, batch_size=1024)
    imgs, labels = next(iter(small_loader))
    with torch.no_grad():
        layer_index = 0
        imgs = imgs.to(device)
        imgs = imgs.view(imgs.size(0), -1)
        # We need to manually loop through the layers to save all activations
        for layer_index, layer in enumerate(net.layers[:-1]):
            imgs = layer(imgs)
            activations[layer_index] = imgs.view(-1).cpu().numpy()

    ## Plotting
    columns = 4
    rows = math.ceil(len(activations)/columns)
    fig, ax = plt.subplots(rows, columns, figsize=(columns*2.7, rows*2.5))
    fig_index = 0
    for key in activations:
        key_ax = ax[fig_index//columns][fig_index%columns]
        sns.histplot(data=activations[key], bins=50, ax=key_ax, color=color, kde=True, stat="density")
        key_ax.set_title(f"Layer {key} - {net.layers[key].__class__.__name__}")
        fig_index += 1
    fig.suptitle(f"Activation distribution for activation function {net.config['act_fn']['name']}", fontsize=14)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.show()
    plt.close()

if __name__ == "__main__":
    ## Create a plot for every activation function
    # for i, act_fn_name in enumerate(act_fn_by_name):
    #     set_seed(
    #         42
    #     )  # Setting the seed ensures that we have the same weight initialization for each activation function
    #     act_fn = act_fn_by_name[act_fn_name]()
    #     net_actfn = BaseNetwork(act_fn=act_fn).to(device)
    #     visualize_gradients(net_actfn, color=f"C{i}")

    for i, act_fn_name in enumerate(act_fn_by_name):
        net_actfn = load_model(model_path=CHECKPOINT_PATH, model_name=f"FashionMNIST_{act_fn_name}").to(device)
        visualize_activations(net_actfn, color=f"C{i}")
