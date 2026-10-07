import os

from huggingface_hub import HfApi

api = HfApi()

# This command recursively pushes everything in your local directory 
# and handles large video files automatically.
# Excludes files with "labeled" in the filename
api.upload_folder(
    folder_path="./export",
    repo_id="jpcurrea/super-fly",
    repo_type="dataset",
    token=os.environ["HF_TOKEN"],
    ignore_patterns=["*labeled*"]
)
