#!/bin/bash

# YOU MUST HAVE A models.txt file
# NEEDS SPACE AT END
models_file="models.txt"

# Check if the file exists
if [[ ! -f "$models_file" ]]; then
  echo "Error: $models_file does not exist."
  exit 1
fi

# Loop through each line (model) in the file
while IFS= read -r model; do
  # Skip empty lines or lines starting with #
  [[ -z "$model" || "$model" == \#* ]] && continue

  echo "Running model: $model"
  python ../main.py --batch-size 5 --repo "$model" --target /mnt/id3/ModelsArchive/
  
  # Check if the command succeeded
  if [[ $? -ne 0 ]]; then
    echo "Error: Failed to process model $model. Exiting."
    exit 1
  fi
done < "$models_file"

echo "All models processed successfully."
