import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision

from dataset import train_dataset


def compare_imgs(img1, img2, title_prefix=""):
    # Calculate MSE loss between both images
    loss = F.mse_loss(img1, img2, reduction="sum")
    # Plot images for visual comparison
    grid = torchvision.utils.make_grid(
        torch.stack([img1, img2], dim=0), nrow=2, normalize=True, value_range=(-1, 1)
    )
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(4, 2))
    plt.title(f"{title_prefix} Loss: {loss.item():4.2f}")
    plt.imshow(grid)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    for i in range(2):
        # Load example image
        img, _ = train_dataset[i]
        img_mean = img.mean(dim=[1, 2], keepdims=True)

        # Shift image by one pixel
        SHIFT = 1
        img_shifted = torch.roll(img, shifts=SHIFT, dims=1)
        img_shifted = torch.roll(img_shifted, shifts=SHIFT, dims=2)
        img_shifted[:, :1, :] = img_mean
        img_shifted[:, :, :1] = img_mean
        compare_imgs(img, img_shifted, "Shifted -")

        # Set half of the image to zero
        img_masked = img.clone()
        img_masked[:, : img_masked.shape[1] // 2, :] = img_mean
        compare_imgs(img, img_masked, "Masked -")
