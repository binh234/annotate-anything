import os
import subprocess

from config import *
from utils import download_file_hf

if __name__ == "__main__":
    if not os.path.exists(sam_checkpoint):
        result = subprocess.run(["wget", sam_url], check=True)
        print(f"wget sam_vit_h_4b8939.pth result = {result}")

    if not os.path.exists(tag2text_checkpoint):
        result = subprocess.run(["wget", tag2text_url], check=True)
        print(f"wget sam_vit_h_4b8939.pth result = {result}")

    if not os.path.exists(dino_config_file):
        download_file_hf(
            repo_id=dino_repo_id, filename=dino_config_file, cache_dir="./"
        )

    if not os.path.exists(dino_checkpoint):
        download_file_hf(repo_id=dino_repo_id, filename=dino_checkpoint, cache_dir="./")
