#!/bin/bash -x

# Provide the full path to the prediction tif/tiff file.
PREDICTION_FILE=${1:-/mnt/dive/shared/zhengyang/label-free/dna/results/gvtnet_label-free/checkpoint_pretrained/prediction.tiff}
# Provide the full path to the target tif/tiff file.
TARGET_FILE=${2:-/mnt/dive/shared/zhengyang/label-free/dna/datasets/test/ground_truth/target.tiff}

# predict
python evaluate.py \
		--prediction_file "$PREDICTION_FILE" \
		--target_file "$TARGET_FILE"
