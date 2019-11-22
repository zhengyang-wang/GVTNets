#!/bin/bash -x

PREDICTION_DIR="results/Planaria_cropped/condition_1/checkpoint_25000/"
TARGET_DIR="../CSBDeep/data/Denoising_Planaria/test_data/GT/"
STATS_FILE="results/Planaria_cropped/checkpoint_25000/stats.pkl"

python evaluate.py \
        --prediction_dir ${PREDICTION_DIR} \
        --target_dir ${TARGET_DIR} \
        --stats_file ${STATS_FILE}