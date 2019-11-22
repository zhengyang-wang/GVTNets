#!/bin/bash -x

RAW_TEST_DATASET_DIR="../CSBDeep/data/Denoising_Tribolium/test_data/"

TIFF_DATASET_DIR="../CSBDeep/data/Denoising_Tribolium/test_data/condition_1/"
RESULT_DIR="results/Tribolium/condition_1/"
MODEL_DIR="saved_models/Tribolium/"
NUM_TEST_PAIRS=20
CHECKPOINT_NUM="25000"
GPU_ID=-1

python datasets/care/format_data.py \
        --raw_test_dataset_dir ${RAW_TEST_DATASET_DIR}

python predict.py \
        --tiff_dataset_dir ${TIFF_DATASET_DIR} \
        --gpu_id ${GPU_ID} \
        --result_dir ${RESULT_DIR} \
        --num_test_pairs ${NUM_TEST_PAIRS} \
        --model_dir ${MODEL_DIR} \
        --checkpoint_num ${CHECKPOINT_NUM} \
        --offset \
        --CARE_normalize