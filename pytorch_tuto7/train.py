import os
import matplotlib.pyplot as plt
import torchvision

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch

from dataset import test_loader, train_loader
from deep_energy_model import DeepEnergyModel
from generative_callback import GenerateCallback
from outlier_callback import OutlierCallback
from sampler_callback import SamplerCallback
from utils import CHECKPOINT_PATH, device


def train_model(**kwargs):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "MNIST"),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=60,
        gradient_clip_val=0.1,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True, mode="min", monitor="val_contrastive_divergence"
            ),
            GenerateCallback(every_n_epochs=5),
            SamplerCallback(every_n_epochs=5),
            OutlierCallback(),
            LearningRateMonitor("epoch"),
        ],
    )
    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "MNIST.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = DeepEnergyModel.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)
        model = DeepEnergyModel(**kwargs)
        trainer.fit(model, train_loader, test_loader)
        model = DeepEnergyModel.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )
    # No testing as we are more interested in other properties
    return model


if __name__ == "__main__":
    # model = train_model(
    #     img_shape=(1, 28, 28), batch_size=train_loader.batch_size, lr=1e-4, beta1=0.0
    # )
    model = DeepEnergyModel.load_from_checkpoint(
        os.path.join(
            CHECKPOINT_PATH,
            "MNIST/lightning_logs/version_1/checkpoints/epoch=59-step=28080.ckpt",
        )
    )
    model.eval()
    model.to(device)
    pl.seed_everything(43)
    callback = GenerateCallback(batch_size=4, vis_steps=8, num_steps=256)
    imgs_per_step = callback.generate_imgs(model)
    imgs_per_step = imgs_per_step.cpu()

    for i in range(imgs_per_step.shape[1]):
        step_size = callback.num_steps // callback.vis_steps
        imgs_to_plot = imgs_per_step[step_size - 1 :: step_size, i]
        imgs_to_plot = torch.cat([imgs_per_step[0:1, i], imgs_to_plot], dim=0)
        grid = torchvision.utils.make_grid(
            imgs_to_plot,
            nrow=imgs_to_plot.shape[0],
            normalize=True,
            value_range=(-1, 1),
            pad_value=0.5,
            padding=2,
        )
        grid = grid.permute(1, 2, 0)
        plt.figure(figsize=(8, 8))
        plt.imshow(grid)
        plt.xlabel("Generation iteration")
        plt.xticks(
            [
                (imgs_per_step.shape[-1] + 2) * (0.5 + j)
                for j in range(callback.vis_steps + 1)
            ],
            labels=[1] + list(range(step_size, imgs_per_step.shape[0] + 1, step_size)),
        )
        plt.yticks([])
        plt.show()
