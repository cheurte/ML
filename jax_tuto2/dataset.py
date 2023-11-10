# from flax.training import train_state
import torch
import os
import jax
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
# import torchvision
from torchvision.datasets import FashionMNIST
import torchvision
# from torchvision import transforms
import urllib.request
from urllib.error import HTTPError

DATASET_PATH = "./data/"
CHECKPOINT_PATH = "./saved_models/tutorial3_jax/"

# Transformations applied on each image =>
# bring them into a numpy array and normalize between -1 and 1
def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    img = (img / 255.0 - 0.5) / 0.5
    return img


# We need to stack the batch elements as numpy arrays
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def load_dataset():
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    print(f"CHECKING DEVICE : {jax.devices()[0]}")

    # Github URL where saved models are stored for this tutorial
    base_url = (
        "https://raw.githubusercontent.com/phlippe/saved_models/main/JAX/tutorial3/"
    )
    # Files to download
    pretrained_files = [
        "FashionMNIST_elu.config",
        "FashionMNIST_elu.tar",
        "FashionMNIST_leakyrelu.config",
        "FashionMNIST_leakyrelu.tar",
        "FashionMNIST_relu.config",
        "FashionMNIST_relu.tar",
        "FashionMNIST_sigmoid.config",
        "FashionMNIST_sigmoid.tar",
        "FashionMNIST_swish.config",
        "FashionMNIST_swish.tar",
        "FashionMNIST_tanh.config",
        "FashionMNIST_tanh.tar",
    ]
    # Create checkpoint path if it doesn't exist yet
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    # For each file, check whether it already exists. If not, try downloading it.
    for file_name in pretrained_files:
        file_path = os.path.join(CHECKPOINT_PATH, file_name)
        if not os.path.isfile(file_path):
            file_url = base_url + file_name
            print(f"Downloading {file_url}...")
            try:
                urllib.request.urlretrieve(file_url, file_path)
            except HTTPError as e:
                print(f"Something went wrong. {e}")

    # Loading the training dataset.
    # We need to split it into a training and validation part
    train_dataset = FashionMNIST(
        root=DATASET_PATH, train=True, transform=image_to_numpy, download=True
    )
    train_set, val_set = torch.utils.data.random_split(
        train_dataset, [50000, 10000], generator=torch.Generator().manual_seed(42)
    )

    # Loading the test set
    test_set = FashionMNIST(
        root=DATASET_PATH, train=False, transform=image_to_numpy, download=True
    )

    # We define a set of data loaders that we can use for various purposes later.
    # Note that for actually training a model, we will use different data loaders
    # with a lower batch size.
    train_loader = data.DataLoader(
        train_set,
        batch_size=1024,
        shuffle=False,
        drop_last=False,
        collate_fn=numpy_collate,
    )
    val_loader = data.DataLoader(
        val_set,
        batch_size=1024,
        shuffle=False,
        drop_last=False,
        collate_fn=numpy_collate,
    )
    test_loader = data.DataLoader(
        test_set,
        batch_size=1024,
        shuffle=False,
        drop_last=False,
        collate_fn=numpy_collate,
    )
    return (
        train_dataset,
        train_set,
        val_set,
        test_set,
        train_loader,
        val_loader,
        test_loader,
    )


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

    exmp_imgs = [train_set[i][0] for i in range(16)]
    # Organize the images into a grid for nicer visualization
    img_grid = torchvision.utils.make_grid(
        torch.from_numpy(np.stack(exmp_imgs, axis=0))[:, None],
        nrow=4,
        normalize=True,
        pad_value=0.5,
    )
    img_grid = img_grid.permute(1, 2, 0)

    plt.figure(figsize=(8, 8))
    plt.title("FashionMNIST examples")
    plt.imshow(img_grid)
    plt.axis("off")
    plt.show()
    plt.close()
