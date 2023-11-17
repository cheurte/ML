import os
import torch
import matplotlib.pyplot as plt
import torchvision

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from autoencoder import Autoencoder
from dataset import get_train_images, test_loader, train_loader, val_loader
from generate_callback import GenerateCallback
from utils import CHECKPOINT_PATH, device, init


def train_cifar(latent_dim):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, f"cifar10_{latent_dim}"),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=500,
        callbacks=[
            ModelCheckpoint(save_weights_only=True),
            GenerateCallback(get_train_images(8), every_n_epochs=10),
            LearningRateMonitor("epoch"),
        ],
    )
    trainer.logger._log_graph = (
        True  # If True, we plot the computation graph in tensorboard
    )
    trainer.logger._default_hp_metric = (
        None  # Optional logging argument that we don't need
    )

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"cifar10_{latent_dim}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = Autoencoder.load_from_checkpoint(pretrained_filename)
    else:
        model = Autoencoder(base_channel_size=32, latent_dim=latent_dim)
        trainer.fit(model, train_loader, val_loader)
    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}
    return model, result

def visualize_reconstructions(model, input_imgs):
    # Reconstruct images
    model.eval()
    with torch.no_grad():
        reconst_imgs = model(input_imgs.to(model.device))
    reconst_imgs = reconst_imgs.cpu()

    # Plotting
    imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0,1)
    grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True, value_range=(-1,1))
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(7,4.5))
    plt.title(f"Reconstructed from {model.hparams.latent_dim} latents")
    plt.imshow(grid)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    init()
    model_dict = {}
    for latent_dim in [64, 128, 256, 384]:
        model_ld, result_ld = train_cifar(latent_dim)
        model_dict[latent_dim] = {"model": model_ld, "result": result_ld}
    latent_dims = sorted([k for k in model_dict])
    val_scores = [model_dict[k]["result"]["val"][0]["test_loss"] for k in latent_dims]

    fig = plt.figure(figsize=(6, 4))
    plt.plot(
        latent_dims,
        val_scores,
        "--",
        color="#000",
        marker="*",
        markeredgecolor="#000",
        markerfacecolor="y",
        markersize=16,
    )
    plt.xscale("log")
    plt.xticks(latent_dims, labels=latent_dims)
    plt.title("Reconstruction error over latent dimensionality", fontsize=14)
    plt.xlabel("Latent dimensionality")
    plt.ylabel("Reconstruction error")
    plt.minorticks_off()
    plt.ylim(0, 100)
    plt.show()

    input_imgs = get_train_images(4)
    for latent_dim in model_dict:
        visualize_reconstructions(model_dict[latent_dim]["model"], input_imgs)
