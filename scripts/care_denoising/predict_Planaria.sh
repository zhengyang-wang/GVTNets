#!/bin/bash -x

TIFF_DATASET_DIR="../CSBDeep/data/Denoising_Planaria/test_data/condition_1/"
RESULT_DIR="results/Planaria/condition_1/"
MODEL_DIR="saved_models/Planaria/"
NUM_TEST_PAIRS=20
CHECKPOINT_NUM="25000"
TEST_PATCH_SIZE_D=128
TEST_PATCH_SIZE_H=256
TEST_PATCH_SIZE_W=256
TEST_BATCH_SIZE=1
GPU_ID=0

python predict.py \
        --tiff_dataset_dir ${TIFF_DATASET_DIR} \
        --gpu_id ${GPU_ID} \
        --result_dir ${RESULT_DIR} \
        --num_test_pairs ${NUM_TEST_PAIRS} \
        --model_dir ${MODEL_DIR} \
        --checkpoint_num ${CHECKPOINT_NUM} \
        --offset \
        --CARE_normalize \
        --cropped_prediction \
        --predict_patch_size ${TEST_PATCH_SIZE_D} ${TEST_PATCH_SIZE_H} ${TEST_PATCH_SIZE_W} \
        --test_batch_size ${TEST_BATCH_SIZE}