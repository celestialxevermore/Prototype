#!/bin/bash

# 사용할 데이터셋 목록
datasets="adult"

# few-shot 샘플 수 목록
few_shots="4 8 16 32 64"

# random seeds
seeds="456 789"

# 실행할 baseline 목록 (띄어쓰기로 구분)
baselines="rf lr xgb mlp cat"

# 실험 실행
for dataset in $datasets; do
    for shot in $few_shots; do
        for seed in $seeds; do
            echo "Running experiment for $dataset with $shot few-shot samples and seed $seed"
            python main_ml.py \
                --source_dataset_name $dataset \
                --few_shot $shot \
                --baseline $baselines \
                --random_seed $seed \
                --train_epochs 200 \
                --batch_size 32 \
                --source_lr 0.001 \
                --dropout_rate 0.3 \
                --threshold 0.5
        done
    done
done