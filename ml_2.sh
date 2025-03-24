#!/bin/bash

# 사용할 데이터셋 목록
gpu_id=1
datasets="heart"

# few-shot 샘플 수 목록
few_shots="4 8 16 32 64"

# random seeds
seed="1993"
baselines="rf lr xgb mlp cat"

# 실험 실행
for dataset in $datasets; do
    for shot in $few_shots; do
        echo "Running experiment for $dataset with $shot few-shot samples and seed $seed"
        CUDA_VISIBLE_DEVICES=$gpu_id python main_ml.py \
        --source_dataset_name $dataset \
        --few_shot $shot \
        --baseline $baselines \
        --base_dir 'ML_results_20250323' \
        --random_seed $seed \
        --train_epochs 200 \
        --batch_size 32 \
        --learning_rate 0.001 \
        --dropout_rate 0.3 \
        --threshold 0.5

    done
done