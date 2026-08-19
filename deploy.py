import os
from huggingface_hub import HfApi

TOKEN = os.environ["HF_TOKEN"]
SPACE = "imanandshah/engine-maintenance-app"

api = HfApi(token=TOKEN)
# NOTE: the Space already exists (grandfathered free Docker Space), so we do NOT
# call create_repo — creating a new Docker Space now requires PRO. We only push files.
for local, remote in [("space/app.py", "app.py"),
                      ("space/preprocessing.py", "preprocessing.py"),
                      ("space/requirements.txt", "requirements.txt"),
                      ("space/Dockerfile", "Dockerfile"),
                      ("space/README.md", "README.md")]:
    api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                    repo_id=SPACE, repo_type="space")
    print("deployed:", remote)
print("Space redeployed ->", SPACE)
