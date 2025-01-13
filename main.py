import argparse
import os
import threading
from huggingface_hub import HfApi, hf_hub_download

# Define default download location and batch size
DEFAULT_DL_LOC = "/mnt/id3/ModelsArchive"
DEFAULT_BATCH_SIZE = 2

# Define ANSI colors for styled output
YELLOW = "\033[93m"
CYAN = "\033[96m"
GREEN = "\033[92m"
GREEN_BOLD = "\033[92;1m"
RESET = "\033[0m"

# Argument parser setup
parser = argparse.ArgumentParser(description="Download all files from a Hugging Face repository in batches.")
parser.add_argument("--repo", required=True, help="The repository ID (e.g., StephanST/WALDO30 or stabilityai/stable-diffusion-3.5-large).")
parser.add_argument("--target", default=DEFAULT_DL_LOC, help=f"The target folder to save the repository (default: {DEFAULT_DL_LOC}).")
parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"The number of files to download per batch (default: {DEFAULT_BATCH_SIZE}).")
args = parser.parse_args()

# Use argument values directly
target_loc = args.target
batch_size = args.batch_size

print(
    f"""\n\n{CYAN}
        ░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓██████▓▒░ ░▒▓██████▓▒░░▒▓████████▓▒░░▒▓██████▓▒░░▒▓███████▓▒░ ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓███████▓▒░  
        ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░ 
        ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░ 
        ░▒▓████████▓▒░▒▓██████▓▒░░▒▓████████▓▒░▒▓█▓▒░      ░▒▓██████▓▒░ ░▒▓████████▓▒░▒▓███████▓▒░░▒▓█▓▒░      ░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒▒▓█▓▒░░▒▓██████▓▒░ ░▒▓███████▓▒░  
        ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░ ░▒▓█▓▓█▓▒░ ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░ 
        ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░ ░▒▓█▓▓█▓▒░ ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░ 
        ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░  ░▒▓██▓▒░  ░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░ 
                                                                                                                                                                                                                                                       
                                                      
    {RESET}
    """
)

# Print download details
print(
    f"\n{YELLOW}Downloading all files from - {CYAN}MODEL: {GREEN}{args.repo} "
    f"{CYAN}LOC: {GREEN}{target_loc} "
    f"{CYAN}BATCH SIZE: {GREEN}{batch_size}{RESET}\n"
)

# Ensure the target directory exists
os.makedirs(target_loc, exist_ok=True)

# Initialize the Hugging Face API
api = HfApi()

# Get the list of files in the repository
files = api.list_repo_files(args.repo)

# Define the function to download a batch of files
def download_batch(batch):
    for file in batch:
        try:
            # Download each file in the current batch
            local_path = hf_hub_download(repo_id=args.repo, filename=file, cache_dir=target_loc)
            print(f"{GREEN}Downloaded: {file} to {local_path}{RESET}")
        except Exception as e:
            print(f"{YELLOW}Failed to download {file}: {e}{RESET}")

# Download files in batches, using threading
if not files:
    print(f"{YELLOW}No files found in the repository: {args.repo}{RESET}")
else:
    # Process files in batches
    threads = []
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]  # Select a batch of files
        print(f"{CYAN}Processing batch: {batch}{RESET}")
        
        # Start a thread to download the batch
        thread = threading.Thread(target=download_batch, args=(batch,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

print(f"{GREEN_BOLD}\n\n    All files downloaded successfully!{RESET}\n")