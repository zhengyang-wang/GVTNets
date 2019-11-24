#!/bin/bash -x

# Provide the dataset name.
DATASET=${1:-dna}
# Provide the name of your model. Use 'gvtnet_label-free_pretrained' for provided pretrained model.
MODEL_NAME=${2:-gvtnet_label-free}
# Provide the number of saved checkpoint. Use 'pretrained' for provided pretrained checkpoint.
CHECKPOINT_NUM=${3:-75000}
# Provide the path to the main folder that saves transformed datasets, checkpoints and results.
SAVE_DIR="save_dir/label-free/${DATASET}"

PREDICTION_DIR="${SAVE_DIR}/results/${MODEL_NAME}/checkpoint_${CHECKPOINT_NUM}/"
TARGET_DIR="${SAVE_DIR}/datasets/test/ground_truth"
STATS_FILE="${SAVE_DIR}/results/stats_${MODEL_NAME}_checkpoint_${CHECKPOINT_NUM}.pkl"

# predict
python evaluate.py \
		--prediction_dir ${PREDICTION_DIR} \
		--target_dir ${TARGET_DIR} \
		--stats_file ${STATS_FILE}
