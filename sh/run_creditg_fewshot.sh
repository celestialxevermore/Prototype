#!/bin/bash

# 실험 횟수 설정
NUM_EXPERIMENTS=100

# 고정된 시드 설정
fixed_seed=42

# 데이터셋 이름 설정
DATASET_NAME="credit-g"

# 실험 결과가 저장될 디렉토리 (이미 존재한다고 가정)
RESULTS_DIR="experiments/${DATASET_NAME}"

for i in $(seq 1 $NUM_EXPERIMENTS)
do
    echo "Running experiment $i with fixed seed: $fixed_seed"
    
    # Python 스크립트 실행
    python main.py \
        --dataset_name ${DATASET_NAME} \
        --dataset_shot 16 \
        --random_seed $fixed_seed \
        --fewshot_epochs 100 \
        --fewshot_lr 0.0001 \
        --batch_size 32 \
        --use_gpu True \
        > ${RESULTS_DIR}/experiment_${i}.log 2>&1
    
    echo "Experiment $i completed"
done

echo "All experiments completed"