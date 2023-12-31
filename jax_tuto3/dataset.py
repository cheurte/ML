import os
import urllib.request
from urllib.error import HTTPError
import torch
import torch.utils.data as data
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import numpy as np


# Github URL where saved models are stored for this tutorial

# Path to the folder where the datasets are/should be downloaded (e.g. MNIST)
DATASET_PATH = "./data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "./saved_models/tutorial4_jax"

base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/JAX/tutorial4/"
# Files to download
pretrained_files = [
    "FashionMNIST_SGD.config",
    "FashionMNIST_SGD_results.json",
    "FashionMNIST_SGD.tar",
    "FashionMNIST_SGDMom.config",
    "FashionMNIST_SGDMom_results.json",
    "FashionMNIST_SGDMom.tar",
    "FashionMNIST_Adam.config",
    "FashionMNIST_Adam_results.json",
    "FashionMNIST_Adam.tar",
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
            print(f"Something went wrong.{e}")


# Transformations applied on each image => bring them into a numpy array and
# normalize to mean 0 and std 1
def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    img = (img / 255.0 - 0.2861) / 0.3530
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


# Loading the training dataset. We need to split it into a training and validation part
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
    train_set, batch_size=1024, shuffle=False, drop_last=False, collate_fn=numpy_collate
)
val_loader = data.DataLoader(
    val_set, batch_size=1024, shuffle=False, drop_last=False, collate_fn=numpy_collate
)
test_loader = data.DataLoader(
    test_set, batch_size=1024, shuffle=False, drop_last=False, collate_fn=numpy_collate
)

if __name__ == "__main__":
    print("Mean", (train_dataset.data.float() / 255.0).mean().item())
    print("Std", (train_dataset.data.float() / 255.0).std().item())
    imgs, _ = next(iter(train_loader))
    print(f"Mean: {imgs.mean().item():5.3f}")
    print(f"Standard deviation: {imgs.std().item():5.3f}")
    print(f"Maximum: {imgs.max().item():5.3f}")
    print(f"Minimum: {imgs.min().item():5.3f}")
