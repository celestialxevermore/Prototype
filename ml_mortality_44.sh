#!/bin/bash

# 사용할 데이터셋 목록
gpu_id=4
#datasets="mimic_mortality eicu_mortality hirid_mortality support_mortality zigong_mortality sic_mortality"
datasets="eicu_mortality"

# few-shot 샘플 수 목록 (full-only 모드라 현재 비활성)
# few_shots="4 8 16 32 64"

# random seeds
#seeds="2095 3192 3155 67 1045"
seeds="44"
baselines="rf lr xgb mlp"
#baselines="cat"
# 실험 실행 — FULL only (--skip_few)
for dataset in $datasets; do
    for seed in $seeds; do
        echo "Running FULL experiment for $dataset with seed $seed"
        CUDA_VISIBLE_DEVICES=$gpu_id python main_ml.py \
        --source_data $dataset \
        --few_shot 4 \
        --baseline $baselines \
        --random_seed $seed \
        --skip_few \
        --train_epochs 200 \
        --batch_size 32 \
        --dropout_rate 0.3 \
        --threshold 0.5 \
        --base_dir test20260424_ML_ICU_full
    done
done