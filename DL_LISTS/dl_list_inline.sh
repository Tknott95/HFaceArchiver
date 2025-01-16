#!/bin/sh

models=(
  "stabilityai/stable-audio-open-1.0"
  "stabilityai/stable-point-aware-3d"
  "google/codegemma-7b"
)

base_command="python ../main.py --batch-size 3 --target /mnt/id3/ModelsArchive/"

for model in "${models[@]}"; do
  echo "Running model: $model"
  eval "$base_command --repo \"$model\""
  if [ $? -ne 0 ]; then
    echo "Error running model: $model. Exiting."
    exit 1
  fi
  echo "Finished model: $model"
done

echo "All models have been processed."
