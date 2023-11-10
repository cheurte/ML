import os

from flax.training import train_state
import jax
from jax import random
import matplotlib.pyplot as plt
import optax
import torch
import torch.utils.data as data
from tqdm import tqdm

from dataset import load_dataset, numpy_collate,CHECKPOINT_PATH
from activation_fct import act_fn_by_name
from network import BaseNetwork
from utils import _get_model_file, load_model, save_model


def calculate_loss(params, apply_fn, batch):
    imgs, labels = batch
    logits = apply_fn(params, imgs)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    acc = (labels == logits.argmax(axis=-1)).mean()
    return loss, acc


@jax.jit
def train_step(state, batch):
    grad_fn = jax.value_and_grad(calculate_loss, has_aux=True)
    (_, acc), grads = grad_fn(state.params, state.apply_fn, batch)
    state = state.apply_gradients(grads=grads)
    return state, acc


@jax.jit
def eval_step(state, batch):
    _, acc = calculate_loss(state.params, state.apply_fn, batch)
    return acc


def train_model(
    net, model_name, max_epochs=50, patience=7, batch_size=256, overwrite=False
):
    """
    Train a model on the training set of FashionMNIST

    Inputs:
        net - Object of BaseNetwork
        model_name - (str) Name of the model, used for creating the checkpoint names
        max_epochs - Number of epochs we want to (maximally) train for
        patience - If the performance on the validation set has not improved for
                #patience epochs, we stop training early
        batch_size - Size of batches used in training
        overwrite - Determines how to handle the case when there already exists
                a checkpoint. If True, it will be overwritten.
                Otherwise, we skip training.
    """
    (
        _,
        train_set,
        _,
        _,
        _,
        val_loader,
        test_loader,
    ) = load_dataset()

    file_exists = os.path.isfile(_get_model_file(CHECKPOINT_PATH, model_name))
    if file_exists and not overwrite:
        print("Model file already exists. Skipping training...")
        state = None
    else:
        if file_exists:
            print("Model file exists, but will be overwritten...")

        # Initializing parameters and training state
        params = net.init(random.PRNGKey(42), exmp_batch[0])
        state = train_state.TrainState.create(
            apply_fn=net.apply,
            params=params,
            tx=optax.sgd(learning_rate=1e-2, momentum=0.9),
        )

        # Defining data loader
        train_loader_local = data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=numpy_collate,
            generator=torch.Generator().manual_seed(42),
        )

        val_scores = []
        best_val_epoch = -1
        for epoch in range(max_epochs):
            ############
            # Training #
            ############
            train_acc = 0.0
            for batch in tqdm(train_loader_local, desc=f"Epoch {epoch+1}", leave=False):
                state, acc = train_step(state, batch)
                train_acc += acc
            train_acc /= len(train_loader_local)

            ##############
            # Validation #
            ##############
            val_acc = test_model(state, val_loader)
            val_scores.append(val_acc)
            print(
                f"[Epoch {epoch+1:2d}] Training accuracy: {train_acc:05.2%},\
                Validation accuracy: {val_acc:4.2%}"
            )

            if len(val_scores) == 1 or val_acc > val_scores[best_val_epoch]:
                print("\t   (New best performance, saving model...)")
                save_model(net, state.params, CHECKPOINT_PATH, model_name)
                best_val_epoch = epoch
            elif best_val_epoch <= epoch - patience:
                print(
                    f"Early stopping due to no improvement over the last \
                    {patience} epochs"
                )
                break

        # Plot a curve of the validation accuracy
        plt.plot([i for i in range(1, len(val_scores) + 1)], val_scores)
        plt.xlabel("Epochs")
        plt.ylabel("Validation accuracy")
        plt.title(f"Validation performance of {model_name}")
        plt.show()
        plt.close()

    state, _ = load_model(CHECKPOINT_PATH, model_name, state=state)
    test_acc = test_model(state, test_loader)
    print((f" Test accuracy: {test_acc:4.2%} ").center(50, "=") + "\n")
    return state, test_acc


def test_model(state, data_loader):
    """
    Test a model on a specified dataset.

    Inputs:
        state - Training state including parameters and model apply function.
        data_loader - DataLoader object of the dataset to test on (validation or test)
    """
    true_preds, count = 0.0, 0
    for batch in data_loader:
        acc = eval_step(state, batch)
        batch_size = batch[0].shape[0]
        true_preds += acc * batch_size
        count += batch_size
    test_acc = true_preds / count
    return test_acc


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
    for act_fn_name in act_fn_by_name:
        print(f"Training BaseNetwork with {act_fn_name} activation...")
        act_fn = act_fn_by_name[act_fn_name]()
        net_actfn = BaseNetwork(act_fn=act_fn)
        train_model(net_actfn, f"FashionMNIST_{act_fn_name}", overwrite=False)
