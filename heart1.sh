#!/bin/bash
gpu_id=0
export CUDA_VISIBLE_DEVICES=$gpu_id
random_seeds="42"
few_shots="4 8 16 32 64"
num_basis_layers="2"
num_basis_heads="6"

# 시나리오 1: heart_target_1 heart_target_2 heart_target_3 heart_target_4 -> heart
echo "=== 시나리오 1: 'Heart_disease_statlog', 'Cardiovascular_Disease_Dataset', 'heart_target_3', 'heart_target_4' -> heart (diversifying_loss on/off) ==="

for random_seed in $random_seeds; do
    for few_shot in $few_shots; do
        
        # Case 1: diversifying_loss OFF
        base_dir="test20251101_divloss_ON"
        echo "Running experiment (div_loss=ON) - seed:${random_seed}, few_shot:${few_shot}"
        CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_S.py \
            --random_seed $random_seed \
            --source_data Heart_disease_statlog Cardiovascular_Disease_Dataset heart_target_3 heart_target_4 \
            --target_data heart \
            --base_dir $base_dir \
            --few_shot $few_shot \
            --num_basis_heads $num_basis_heads \
            --num_basis_layers $num_basis_layers \
            --diversifying_loss \
            --des USE_Diversifying_Loss_MultiSource
    done
done

echo "=== 모든 실험 완료 ==="