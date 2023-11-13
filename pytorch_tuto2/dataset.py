import os
import matplotlib.pyplot as plt
import torch.utils.data as data
from urllib.error import HTTPError
import urllib.request

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import FashionMNIST

# Path to the folder where the datasets are/should be downloaded (e.g. MNIST)
DATASET_PATH = "./data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "./saved_models/tutorial3"

device = (
    torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
)


# Function for setting the seed
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(42)

# Additionally, some operations on a GPU are implemented stochastic for efficiency
# We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Transformations applied on each image => first make them a tensor, then normalize them in the range -1 to 1
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# Loading the training dataset. We need to split it into a training and validation part
train_dataset = FashionMNIST(
    root=DATASET_PATH, train=True, transform=transform, download=True
)
train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])

# Loading the test set
test_set = FashionMNIST(
    root=DATASET_PATH, train=False, transform=transform, download=True
)

# We define a set of data loaders that we can use for various purposes later.
# Note that for actually training a model, we will use different data loaders
# with a lower batch size.
train_loader = data.DataLoader(
    train_set, batch_size=1024, shuffle=True, drop_last=False
)
val_loader = data.DataLoader(val_set, batch_size=1024, shuffle=False, drop_last=False)
test_loader = data.DataLoader(test_set, batch_size=1024, shuffle=False, drop_last=False)

if __name__ == "__main__":
    # Github URL where saved models are stored for this tutorial
    base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial3/"
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
                print(f"Something went wrong{e}")

        print("Using device", device)
    exmp_imgs = [train_set[i][0] for i in range(16)]
    # Organize the images into a grid for nicer visualization
    img_grid = torchvision.utils.make_grid(
        torch.stack(exmp_imgs, dim=0), nrow=4, normalize=True, pad_value=0.5
    )
    img_grid = img_grid.permute(1, 2, 0)

    plt.figure(figsize=(8, 8))
    plt.title("FashionMNIST examples")
    plt.imshow(img_grid)
    plt.axis("off")
    plt.show()
    plt.close()
