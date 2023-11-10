import jax
from jax import random
import optax
import seaborn as sns
import matplotlib.pyplot as plt
import torch.utils.data as data
from dataset import load_dataset, numpy_collate
from activation_fct import act_fn_by_name
from network import BaseNetwork

def visualize_gradients(net, params, color="C0"):
    """
    Inputs:
        net - Object of class BaseNetwork
        color - Color in which we want to visualize the histogram (for easier
            separation of activation functions)
    """

    # Pass one batch through the network, and calculate the gradients for the weights
    def loss_func(p):
        imgs, labels = exmp_batch
        logits = net.apply(p, imgs)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        return loss

    grads = jax.grad(loss_func)(params)
    grads = jax.device_get(grads)
    # We limit our visualization to the weight parameters and exclude the bias
    # to reduce the number of plots
    grads = jax.tree_util.tree_leaves(grads)
    grads = [g.reshape(-1) for g in grads if len(g.shape) > 1]

    ## Plotting
    columns = len(grads)
    fig, ax = plt.subplots(1, columns, figsize=(columns * 3.5, 2.5))
    # fig_index = 0
    for g_idx, g in enumerate(grads):
        key = f"Layer {g_idx * 2} - weights"
        key_ax = ax[g_idx % columns]
        sns.histplot(data=g, bins="auto", ax=key_ax, color=color, kde=True)
        key_ax.set_title(str(key))
        key_ax.set_xlabel("Grad magnitude")
    fig.suptitle(
        f"Gradient magnitude distribution for activation function \
        {net.act_fn.__class__.__name__}",
        fontsize=14,
        y=1.05,
    )
    fig.subplots_adjust(wspace=0.45)
    plt.show()
    plt.close()


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
    for i, act_fn_name in enumerate(act_fn_by_name):
        act_fn = act_fn_by_name[act_fn_name]()
        net_actfn = BaseNetwork(act_fn=act_fn)
        params = net_actfn.init(random.PRNGKey(0), exmp_batch[0])
        visualize_gradients(net_actfn, params, color=f"C{i}")
