import os
import pytorch_lightning as pl
import torch
from utils import CHECKPOINT_PATH
from dataset import train_loader, test_loader
import torchvision
import matplotlib.pyplot as plt
from deep_energy_model import DeepEnergyModel


@torch.no_grad()
def compare_images(img1, img2):
    imgs = torch.stack([img1, img2], dim=0).to(model.device)
    score1, score2 = model.cnn(imgs).cpu().chunk(2, dim=0)
    grid = torchvision.utils.make_grid(
        [img1.cpu(), img2.cpu()],
        nrow=2,
        normalize=True,
        value_range=(-1, 1),
        pad_value=0.5,
        padding=2,
    )
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(4, 4))
    plt.imshow(grid)
    plt.xticks(
        [(img1.shape[2] + 2) * (0.5 + j) for j in range(2)],
        labels=["Original image", "Transformed image"],
    )
    plt.yticks([])
    plt.show()
    print(f"Score original image: {score1.item():4.2f}")
    print(f"Score transformed image: {score2.item():4.2f}")


if __name__ == "__main__":
    model = DeepEnergyModel.load_from_checkpoint(
        os.path.join(
            CHECKPOINT_PATH,
            "MNIST/lightning_logs/version_1/checkpoints/epoch=59-step=28080.ckpt",
        )
    )

    # Setting the seed
    pl.seed_everything(43)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        rand_imgs = torch.rand((128,) + model.hparams.img_shape).to(model.device)
        rand_imgs = rand_imgs * 2 - 1.0
        rand_out = model.cnn(rand_imgs).mean()
        print(f"Average score for random images: {rand_out.item():4.2f}")

    with torch.no_grad():
        train_imgs, _ = next(iter(train_loader))
        train_imgs = train_imgs.to(model.device)
        train_out = model.cnn(train_imgs).mean()
        print(f"Average score for training images: {train_out.item():4.2f}")

    test_imgs, _ = next(iter(test_loader))
    exmp_img = test_imgs[0].to(model.device)

    img_noisy = exmp_img + torch.randn_like(exmp_img) * 0.3
    img_noisy.clamp_(min=-1.0, max=1.0)
    compare_images(exmp_img, img_noisy)

    img_flipped = exmp_img.flip(dims=(1, 2))
    compare_images(exmp_img, img_flipped)

    img_tiny = torch.zeros_like(exmp_img) - 1
    img_tiny[:, exmp_img.shape[1] // 2 :, exmp_img.shape[2] // 2 :] = exmp_img[
        :, ::2, ::2
    ]
    compare_images(exmp_img, img_tiny)
