from huggingface_hub import HfApi
import os

api = HfApi()

# Delete labeled files from the repo
fns = os.listdir("./export")
unlabeled_files = [fn for fn in fns if "labeled" not in fn]

for file in unlabeled_files:
    api.delete_file(
        path_in_repo=file,
        repo_id="jpcurrea/super-fly",
        repo_type="dataset",
        token=os.environ["HF_TOKEN"]
    )
