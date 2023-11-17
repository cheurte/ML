import urllib.request
from urllib.error import HTTPError
import os
from utils import CHECKPOINT_PATH, DATASET_PATH

import torch_geometric

cora_dataset = torch_geometric.datasets.Planetoid(root=DATASET_PATH, name="Cora")
tu_dataset = torch_geometric.datasets.TUDataset(root=DATASET_PATH, name="MUTAG")
if __name__ == "__main__":
    # Github URL where saved models are stored for this tutorial
    base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial7/"
    # Files to download
    pretrained_files = [
        "NodeLevelMLP.ckpt",
        "NodeLevelGNN.ckpt",
        "GraphLevelGraphConv.ckpt",
    ]

    # Create checkpoint path if it doesn't exist yet
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    # For each file, check whether it already exists. If not, try downloading it.
    for file_name in pretrained_files:
        file_path = os.path.join(CHECKPOINT_PATH, file_name)
        if "/" in file_name:
            os.makedirs(file_path.rsplit("/", 1)[0], exist_ok=True)
        if not os.path.isfile(file_path):
            file_url = base_url + file_name
            print(f"Downloading {file_url}...")
            try:
                urllib.request.urlretrieve(file_url, file_path)
            except HTTPError as e:
                print(
                    "Something went wrong. Please try to download the file from the GDrive folder, or contact the author with the full output including the following error:\n",
                    e,
                )
    print(cora_dataset[0])
    print("#"*50)
    print("#"*50)
    print("Data object:", tu_dataset.data)
    print("Length:", len(tu_dataset))
    print(f"Average label: {tu_dataset.data.y.float().mean().item():4.2f}")
