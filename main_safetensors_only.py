import argparse
from huggingface_hub import HfApi, hf_hub_download
import os

# Define default download location
DEFAULT_DL_LOC = "/mnt/id3/ModelsArchive"

# Define ANSI colors for styled output
YELLOW = "\033[93m"
CYAN = "\033[96m"
GREEN = "\033[92m"
RESET = "\033[0m"

# Define model file extensions to filter
MODEL_EXTENSIONS = {".bin", ".pt", ".h5", ".onnx", ".ckpt", ".safetensors", ".json", ".txt"}

# Argument parser setup
parser = argparse.ArgumentParser(description="Download model files from a Hugging Face repository.")
parser.add_argument("--repo", required=True, help="The repository ID (e.g., StephanST/WALDO30 or stabilityai/stable-diffusion-3.5-large).")
parser.add_argument("--target", help="The target folder to save the repository.")
args = parser.parse_args()

# Resolve target download location
target_loc = args.target if args.target else DEFAULT_DL_LOC

# Print download details
print(
    f"\n{YELLOW}Downloading - {CYAN}MODEL: {GREEN}{args.repo} "
    f"{CYAN}LOC: {GREEN}{target_loc}{RESET}\n"
)

# Ensure the target directory exists
os.makedirs(target_loc, exist_ok=True)

# Initialize Hugging Face API
api = HfApi()

# Fetch file list from the repository
files = api.list_repo_files(repo_id=args.repo)

# Filter files for model extensions
model_files = [file for file in files if any(file.endswith(ext) for ext in MODEL_EXTENSIONS)]

if not model_files:
    print(f"{YELLOW}No model files found in the repository: {args.repo}{RESET}")
else:
    for model_file in model_files:
        # Download each model file
        local_path = hf_hub_download(repo_id=args.repo, filename=model_file, cache_dir=target_loc)
        print(f"{GREEN}Downloaded: {model_file} to {local_path}{RESET}")

