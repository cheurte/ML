import numpy as np
from optimizer import sgd, sgd_momentum, adam
import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp
import optax


def bivar_gaussian(w1, w2, x_mean=0.0, y_mean=0.0, x_sig=1.0, y_sig=1.0):
    norm = 1 / (2 * np.pi * x_sig * y_sig)
    x_exp = (-1 * (w1 - x_mean) ** 2) / (2 * x_sig**2)
    y_exp = (-1 * (w2 - y_mean) ** 2) / (2 * y_sig**2)
    return norm * jnp.exp(x_exp + y_exp)


def comb_func(w1, w2):
    z = -bivar_gaussian(w1, w2, x_mean=1.0, y_mean=-0.5, x_sig=0.2, y_sig=0.2)
    z -= bivar_gaussian(w1, w2, x_mean=-1.0, y_mean=0.5, x_sig=0.2, y_sig=0.2)
    z -= bivar_gaussian(w1, w2, x_mean=-0.5, y_mean=-0.8, x_sig=0.2, y_sig=0.2)
    return z


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
    optimizer, curve_func=bivar_gaussian, num_updates=100, init=[5, 5]
):
    """
    Inputs:
        optimizer - Optimizer to use
        curve_func - Loss function (e.g. pathological curvature)
        num_updates - Number of updates/steps to take when optimizing
        init - Initial values of parameters.
        Must be a list/tuple with two elements representing w_1 and w_2
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
    _ = plot_curve(
        comb_func, x_range=(-2, 2), y_range=(-2, 2), plot_3d=True, title="Steep optima"
    )
    # plt.show()
    SGD_points = train_curve(sgd(lr=0.5), comb_func, init=[0, 0])
    SGDMom_points = train_curve(
        sgd_momentum(lr=1, momentum=0.9), comb_func, init=[0, 0]
    )
    Adam_points = train_curve(adam(lr=0.2), comb_func, init=[0, 0])

    all_points = np.concatenate([SGD_points, SGDMom_points, Adam_points], axis=0)
    ax = plot_curve(
        comb_func, x_range=(-2, 2), y_range=(-2, 2), plot_3d=False, title="Steep optima"
    )
    ax.plot(
        SGD_points[:, 0],
        SGD_points[:, 1],
        color="red",
        marker="o",
        zorder=3,
        label="SGD",
        alpha=0.7,
    )
    ax.plot(
        SGDMom_points[:, 0],
        SGDMom_points[:, 1],
        color="blue",
        marker="o",
        zorder=2,
        label="SGDMom",
        alpha=0.7,
    )
    ax.plot(
        Adam_points[:, 0],
        Adam_points[:, 1],
        color="grey",
        marker="o",
        zorder=1,
        label="Adam",
        alpha=0.7,
    )
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    plt.legend()
    plt.show()
