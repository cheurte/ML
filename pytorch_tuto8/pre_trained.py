import urllib.request
from urllib.error import HTTPError
from utils import CHECKPOINT_PATH, init
import os

init()
if __name__ == "__main__":
    # Github URL where saved models are stored for this tutorial
    base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial9/"
    # Files to download
    pretrained_files = [
        "cifar10_64.ckpt",
        "cifar10_128.ckpt",
        "cifar10_256.ckpt",
        "cifar10_384.ckpt",
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
                print(
                    "Something went wrong. Please try to download the file from the GDrive folder, or contact the author with the full output including the following error:\n",
                    e,
                )
