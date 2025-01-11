import argparse 
from huggingface_hub import snapshot_download

""" EXAMPLE

 with default target
   python script.py --repo <MODEL-NAME>
 with download loc
   python script.py --repo <MODEL-NAME> --target <DOWNLOAD_LOC>
  
"""

DEFAULT_DL_LOC="/mnt/id3/ModelsArchive"

YELLOW = "\033[93m"
CYAN = "\033[96m"
GREEN = "\033[92m"
RESET = "\033[0m"


parser = argparse.ArgumentParser(description="Download a Hugging Face repository snapshot.")
parser.add_argument("--repo", required=True, help="The repository ID (e.g., StephanST/WALDO30 or stabilityai/stable-diffusion-3.5-large) .")
parser.add_argument("--target", required=True, help="The target folder to save the repository.")
args = parser.parse_args()

target_loc = args.target if args.target else DEFAULT_DL_LOC

print(
    f"\n{YELLOW}Downloading - {CYAN}MODEL: {GREEN}{args.repo} "
    f"{CYAN}LOC: {GREEN}{target_loc}{RESET}\n"
)

snapshot_download(repo_id=args.repo, cache_dir=target_loc)