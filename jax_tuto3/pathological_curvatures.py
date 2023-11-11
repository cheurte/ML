import flax.linen as nn
from optimizer import sgd, sgd_momentum, adam
import optax
import seaborn as sns
import jax
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp


def pathological_curve_loss(w1, w2):
    # Example of a pathological curvature.
    # There are many more possible, feel free to experiment here!
    x1_loss = nn.tanh(w1) ** 2 + 0.01 * jnp.abs(w1)
    x2_loss = nn.sigmoid(w2)
    return x1_loss + x2_loss


def plot_curve(
    curve_fn,
    x_range=(-5, 5),
    y_range=(-5, 5),
    plot_3d=False,
    cmap="viridis",
    title="Pathological curvature",
):
    # fig = plt.figure()
    ax = plt.axes(projection="3d") if plot_3d else plt.axes()

    x = np.arange(x_range[0], x_range[1], (x_range[1] - x_range[0]) / 100.0)
    y = np.arange(y_range[0], y_range[1], (y_range[1] - y_range[0]) / 100.0)
    x, y = np.meshgrid(x, y)
    z = curve_fn(x, y)
    z = jax.device_get(z)

    if plot_3d:
        ax.plot_surface(
            x,
            y,
            z,
            cmap=cmap,
            linewidth=1,
            color="#000",
            antialiased=False,
            edgecolor="none",
        )
        ax.set_zlabel("loss")
    else:
        ax.imshow(
            z[::-1], cmap=cmap, extent=(x_range[0], x_range[1], y_range[0], y_range[1])
        )
    plt.title(title)
    ax.set_xlabel(r"$w_1$")
    ax.set_ylabel(r"$w_2$")
    plt.tight_layout()
    return ax


def train_curve(
    optimizer, curve_func=pathological_curve_loss, num_updates=100, init=[5, 5]
):
    """
    Inputs:
        optimizer - Optimizer to use
        curve_func - Loss function (e.g. pathological curvature)
        num_updates - Number of updates/steps to take when optimizing
        init - Initial values of parameters. Must be a list/tuple with two
        elements representing w_1 and w_2
    Outputs:
        Numpy array of shape [num_updates, 3] with [t,:2] being the parameter
        values at step t, and [t,2] the loss at t.
    """
    weights = jnp.array(init, dtype=jnp.float32)
    grad_fn = jax.jit(jax.value_and_grad(lambda w: curve_func(w[0], w[1])))
    opt_state = optimizer.init(weights)

    list_points = []
    for _ in range(num_updates):
        loss, grads = grad_fn(weights)
        list_points.append(jnp.concatenate([weights, loss[None]], axis=0))
        updates, opt_state = optimizer.update(grads, opt_state)
        weights = optax.apply_updates(weights, updates)
    points = jnp.stack(list_points, axis=0)
    points = jax.device_get(points)
    return points


if __name__ == "__main__":
    sns.reset_orig()
    _ = plot_curve(pathological_curve_loss, plot_3d=True)
    plt.show()
    SGD_points = train_curve(sgd(lr=10))
    SGDMom_points = train_curve(sgd_momentum(lr=10, momentum=0.9))
    Adam_points = train_curve(adam(lr=1))
    all_points = np.concatenate([SGD_points, SGDMom_points, Adam_points], axis=0)
    ax = plot_curve(
        pathological_curve_loss,
        x_range=(
            -np.absolute(all_points[:, 0]).max(),
            np.absolute(all_points[:, 0]).max(),
        ),
        y_range=(all_points[:, 1].min(), all_points[:, 1].max()),
        plot_3d=False,
    )
    ax.plot(
        SGD_points[:, 0],
        SGD_points[:, 1],
        color="red",
        marker="o",
        zorder=1,
        label="SGD",
    )
    ax.plot(
        SGDMom_points[:, 0],
        SGDMom_points[:, 1],
        color="blue",
        marker="o",
        zorder=2,
        label="SGDMom",
    )
    ax.plot(
        Adam_points[:, 0],
        Adam_points[:, 1],
        color="grey",
        marker="o",
        zorder=3,
        label="Adam",
    )
    plt.legend()
    plt.show()
