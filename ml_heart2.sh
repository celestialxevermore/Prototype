#!/bin/bash

gpu_id=4
baseline="xgb"
datasets="Heart_disease_statlog Cardiovascular_Disease_Dataset Medicaldataset heart_failure_clinical_records cardio_SAheart Erbil_Cardiovascular_Health_Dataset"
seeds="42 44 46 48 50"
few_shots="4 8 16 32 64"
base_dir="ML_test20251221"

for dataset in $datasets; do
  for shot in $few_shots; do
    for seed in $seeds; do
      echo "Running - dataset:$dataset, baseline:$baseline, few_shot:$shot, seed:$seed"
      CUDA_VISIBLE_DEVICES=$gpu_id python main_ml.py \
        --source_data $dataset \
        --few_shot $shot \
        --baseline $baseline \
        --random_seed $seed \
        --base_dir $base_dir \
        --des ML_${baseline}_20251221
    done
  done
done
