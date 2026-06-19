#!/bin/bash

source ./config.sh

python train_diabetes.py  --alpha $alpha --l1_ratio $l1_ratio --experiment_name "capsule_$CO_CAPSULE_ID" --input_data "$input_data" --s3_bucket $s3_bucket