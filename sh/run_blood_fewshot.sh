#!/bin/bash

# 실험 횟수 설정
NUM_EXPERIMENTS=10000

# 고정된 시드 설정
fixed_seed=42

# 결과를 저장할 디렉토리 생성
mkdir -p blood_fewshot_results

for i in $(seq 1 $NUM_EXPERIMENTS)
do
    echo "Running experiment $i with fixed seed: $fixed_seed"
    
    # Python 스크립트 실행
    python main.py --dataset_name blood --fewshot --dataset_shot 16 --random_seed $fixed_seed > blood_fewshot_results/experiment_${i}.log 2>&1
    
    echo "Experiment $i completed"
done

echo "All experiments completed"
