import math

import matplotlib.pyplot as plt
import seaborn as sns
import torch.utils.data as data

from activation_fct import act_fn_by_name
from dataset import load_dataset
from dataset import CHECKPOINT_PATH, numpy_collate
from utils import load_model


def visualize_activations(net, exmp_batch, color="C0"):
    activations = {}

    imgs, _ = exmp_batch
    _, activations = net(imgs, return_activations=True)

    ## Plotting
    columns = 4
    rows = math.ceil(len(activations) / columns)
    fig, ax = plt.subplots(rows, columns, figsize=(columns * 2.7, rows * 2.5))
    act_fn_name = net.act_fn.__class__.__name__
    for idx, activ in enumerate(activations):
        key_ax = ax[idx // columns][idx % columns]
        sns.histplot(
            data=activ.reshape(-1),
            bins="auto",
            ax=key_ax,
            color=color,
            kde=True,
            stat="density",
        )
        key_ax.set_title(f"Layer {idx} - {'Dense' if idx%2==0 else act_fn_name}")
    fig.suptitle(
        f"Activation distribution for activation function {act_fn_name}", fontsize=14
    )
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.show()
    plt.close()


if __name__ == "__main__":
    (
        _,
        train_set,
        _,
        _,
        _,
        _,
        _,
    ) = load_dataset()
    small_loader = data.DataLoader(
        train_set, batch_size=256, shuffle=False, collate_fn=numpy_collate
    )
    exmp_batch = next(iter(small_loader))

    for i, act_fn_name in enumerate(act_fn_by_name):
        state, net_actfn = load_model(
            model_path=CHECKPOINT_PATH, model_name=f"FashionMNIST_{act_fn_name}"
        )
        if net_actfn is not None:
            net_actfn = net_actfn.bind(state.params)
            visualize_activations(net_actfn, exmp_batch, color=f"C{i}")
