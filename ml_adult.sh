#!/bin/bash

# 사용할 데이터셋 목록
gpu_id=4
datasets="adult"

# few-shot 샘플 수 목록
few_shots="4 8 16 32 64"

# random seeds
seeds="2011 1023 4005 3007 4017"
baselines="rf lr xgb mlp cat"

# 실험 실행
for dataset in $datasets; do
    for shot in $few_shots; do
        for seed in $seeds; do
            echo "Running experiment for $dataset with $shot few-shot samples and seed $seed"
            CUDA_VISIBLE_DEVICES=$gpu_id python main_ml.py \
            --source_data $dataset \
            --few_shot $shot \
            --baseline $baselines \
            --random_seed $seed \
            --train_epochs 200 \
            --batch_size 32 \
            --dropout_rate 0.3 \
            --threshold 0.5 \
            --base_dir test202050721
        done
    done
done