#!/bin/bash -x

DATASET=${1:-dna}
CSV_DATASET_DIR="csvs/${DATASET}/"
RAW_DATASET_DIR="/mnt/dive/shared/yaochen.xie/Label_free_prediction"
NPZ_DATASET_DIR="/mnt/dive/shared/zhengyang/label-free/${DATASET}/datasets"
NUM_TRAIN_PAIRS=30
NUM_TEST_PAIRS=20

# generate datasets
python generate_npz.py \
	--csv_dataset_dir ${CSV_DATASET_DIR} \
	--raw_dataset_dir ${RAW_DATASET_DIR} \
	--npz_dataset_dir ${NPZ_DATASET_DIR} \
	--num_train_pairs ${NUM_TRAIN_PAIRS} \
	--num_test_pairs ${NUM_TEST_PAIRS}
