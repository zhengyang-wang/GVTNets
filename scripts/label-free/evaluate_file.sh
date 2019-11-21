#!/bin/bash -x

PREDICTION_FILE=${1:-prediction.tiff} # Provide the full path to the prediction tif/tiff file.
TARGET_FILE=${2:-target.tiff} # Provide the full path to the target tif/tiff file.

# predict
python evaluate.py \
		--prediction_file "$PREDICTION_FILE" \
		--target_file "$TARGET_FILE"
