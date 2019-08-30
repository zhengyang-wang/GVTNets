#!/bin/bash -x

DATASET=${1:-dna}
SAVE_DIR="/mnt/dive/shared/zhengyang/label-free/${DATASET}"
TEST_PATCH_SIZE_D=0
TEST_PATCH_SIZE_H=0
TEST_PATCH_SIZE_W=0
TEST_STEP_SIZE_D=0
TEST_STEP_SIZE_H=0
TEST_STEP_SIZE_W=0
RESULT_DIR="${SAVE_DIR}/results"
CHECKPOINT_NUM=${3:-100000}
MODEL_NAME=${2:-"model9"}
OVERWRITE="FALSE"

STES_STEP="[${TEST_STEP_SIZE_D}, ${TEST_STEP_SIZE_H}, ${TEST_STEP_SIZE_W}]"
PATCH_STEP="[${TEST_PATCH_SIZE_D}, ${TEST_PATCH_SIZE_H}, ${TEST_PATCH_SIZE_W}]"
PREDICTION_DIR="${SAVE_DIR}/results/$MODEL_NAME/checkpoint_${CHECKPOINT_NUM}/patch"$PATCH_STEP"_step"$STES_STEP""
TARGET_DIR="${SAVE_DIR}/results/"

# predict
python evaluate.py \
		--prediction_dir "$PREDICTION_DIR" \
		--target_dir "$TARGET_DIR" \
		--overwrite "$OVERWRITE"
