import argparse
import os
from huggingface_hub import HfApi, hf_hub_download

# Define default download location
DEFAULT_DL_LOC = "/mnt/id3/ModelsArchive"

# Define ANSI colors for styled output
YELLOW = "\033[93m"
CYAN = "\033[96m"
GREEN = "\033[92m"
RESET = "\033[0m"

# Argument parser setup
parser = argparse.ArgumentParser(description="Download all files from a Hugging Face repository.")
parser.add_argument("--repo", required=True, help="The repository ID (e.g., StephanST/WALDO30 or stabilityai/stable-diffusion-3.5-large).")
parser.add_argument("--target", help="The target folder to save the repository.")
args = parser.parse_args()

# Resolve target download location
target_loc = args.target if args.target else DEFAULT_DL_LOC

# Print download details
print(
    f"\n{YELLOW}Downloading all files from - {CYAN}MODEL: {GREEN}{args.repo} "
    f"{CYAN}LOC: {GREEN}{target_loc}{RESET}\n"
)

# Ensure the target directory exists
os.makedirs(target_loc, exist_ok=True)

# Initialize the Hugging Face API
api = HfApi()

# Get the list of files in the repository
files = api.list_repo_files(args.repo)

# Download all files in the repository
if not files:
    print(f"{YELLOW}No files found in the repository: {args.repo}{RESET}")
else:
    for file in files:
        # Download each file
        local_path = hf_hub_download(repo_id=args.repo, filename=file, cache_dir=target_loc)
        print(f"{GREEN}Downloaded: {file} to {local_path}{RESET}")
