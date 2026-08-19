import os
from huggingface_hub import HfApi

TOKEN = os.environ["HF_TOKEN"]
SPACE = "imanandshah/engine-maintenance-app"

api = HfApi(token=TOKEN)
api.create_repo(SPACE, repo_type="space", space_sdk="docker", exist_ok=True)
for local, remote in [("space/app.py", "app.py"),
                      ("space/preprocessing.py", "preprocessing.py"),
                      ("space/requirements.txt", "requirements.txt"),
                      ("space/Dockerfile", "Dockerfile"),
                      ("space/README.md", "README.md")]:
    api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                    repo_id=SPACE, repo_type="space")
    print("deployed:", remote)
print("Space redeployed ->", SPACE)
