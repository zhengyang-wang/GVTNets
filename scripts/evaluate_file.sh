#!/bin/bash -x

PREDICTION_FILE="/mnt/dive/shared/zhengyang/label-free/fibrillarin/results/model1/checkpoint_99999/patch[0, 0, 0]_step[0, 0, 0]/prediction_00.tiff"
TARGET_FILE="/mnt/dive/shared/zhengyang/label-free/fibrillarin/results/target_00.tiff"

# predict
python evaluate.py \
		--prediction_file "$PREDICTION_FILE" \
		--target_file "$TARGET_FILE"
