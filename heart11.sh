#!/bin/bash
gpu_id=0
export CUDA_VISIBLE_DEVICES=$gpu_id
random_seeds="42"
few_shots="4 8 16 32"
num_basis_layers="2"
num_basis_heads="1"
# 시나리오 1: heart_target_1 heart_target_2 heart_target_3 heart_target_4 -> heart
echo "=== 시나리오 1: 'Heart_disease_statlog', 'Cardiovascular_Disease_Dataset', 'heart_target_3', 'heart_target_4' -> heart_disease (diversifying_loss on/off) ==="

for random_seed in $random_seeds; do
    for few_shot in $few_shots; do
        
        # Case 1: diversifying_loss OFF
        base_dir="test20251103_MultiSource_som_loss_YES_LCG_diversifying_loss_Target_heart_disease_sampling100"
        echo "Running experiment (div_loss=ON) - seed:${random_seed}, few_shot:${few_shot}"
        CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_S.py \
            --random_seed $random_seed \
            --source_data Heart_disease_statlog Cardiovascular_Disease_Dataset heart_target_3 heart_target_4 \
            --target_data hungarian \
            --base_dir $base_dir \
            --few_shot $few_shot \
            --num_basis_heads $num_basis_heads \
            --num_basis_layers $num_basis_layers \
            --diversifying_loss \
            --des MultiSource_YES_LCG_diversifying_loss_20251103
    done
done


echo "=== 모든 실험 완료 ==="