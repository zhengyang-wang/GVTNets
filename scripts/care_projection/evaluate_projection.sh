#!/bin/bash -x

# Provide the condition name.
CONDITION=${1:-C1}
# Provide the name of your model. Use 'gvtnet_care_pretrained' for provided pretrained model.
MODEL_NAME=${2:-"gvtnet_care"}
# Provide the number of saved checkpoint. Use 'pretrained' for provided pretrained model.
CHECKPOINT_NUM=${3:-45000}
# Provide the path to the main folder that saves transformed datasets, checkpoints and results.
SAVE_DIR="save_dir/care_projection/flywing"

PREDICTION_DIR="${SAVE_DIR}/results/${MODEL_NAME}/${CONDITION}/checkpoint_${CHECKPOINT_NUM}/"
TARGET_DIR="/mnt/dive/shared/yaochen.xie/CSBDeep/Projection_Flywing/test_data/PreMosa/GT"
STATS_FILE="${SAVE_DIR}/results/stats_${MODEL_NAME}_checkpoint_${CHECKPOINT_NUM}.pkl"


python evaluate.py \
        --prediction_dir ${PREDICTION_DIR} \
        --target_dir ${TARGET_DIR} \
        --stats_file ${STATS_FILE}