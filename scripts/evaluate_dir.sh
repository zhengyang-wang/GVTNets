#!/bin/bash -x

DATASET=${1:-dna}
CHECKPOINT_NUM=${3:-75000}
MODEL_NAME=${2:-model9}
SAVE_DIR="/mnt/dive/shared/zhengyang/label-free/${DATASET}"
PREDICTION_DIR="${SAVE_DIR}/results/${MODEL_NAME}/checkpoint_${CHECKPOINT_NUM}/"
TARGET_DIR="${SAVE_DIR}/datasets/test/ground_truth"
STATS_FILE="${SAVE_DIR}/results/stats_${MODEL_NAME}_checkpoint_${CHECKPOINT_NUM}.pkl"

# predict
python evaluate.py \
		--prediction_dir ${PREDICTION_DIR} \
		--target_dir ${TARGET_DIR} \
		--stats_file ${STATS_FILE}
