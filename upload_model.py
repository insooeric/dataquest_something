from huggingface_hub import HfApi

api = HfApi(token="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")
username = api.whoami()["name"]
print(f"Logged in as: {username}")

repo_id = f"{username}/woundscope"
api.create_repo(repo_id, repo_type="model", exist_ok=True)
print(f"Repo ready: {repo_id}")

api.upload_file(
    path_or_fileobj="models/woundscope_v3.pth",
    path_in_repo="woundscope_v3.pth",
    repo_id=repo_id,
    repo_type="model",
)
print("done")
