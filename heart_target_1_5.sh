#!/bin/bash
gpu_id=4
#export CUDA_VISIBLE_DEVICES=$gpu_id
random_seeds="42 44 46 48 50"
few_shots="4 8 16 32 64"
num_basis_layers="2"
num_basis_heads="4"
num_graphs="8 10"
echo "=== 시나리오 2 VQVAE Additional FGW: 'Heart_disease_statlog', 'Cardiovascular_Disease_Dataset', 'heart_target_3', 'heart_target_4' -> heart_target_1 ==="
for random_seed in $random_seeds; do
    for n_graphs in $num_graphs; do
        for few_shot in $few_shots; do
            base_dir="test20251106_MultiSource_VQVAE_Additional_FGW_Target_heart_target_1"
            echo "Running experiment - seed:${random_seed}, few_shot:${few_shot}"
            CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_S.py \
                --random_seed $random_seed \
                --source_data Heart_disease_statlog Cardiovascular_Disease_Dataset heart_target_3 heart_target_4 \
                --target_data heart_target_1 \
                --base_dir $base_dir \
                --few_shot $few_shot \
                --n_graphs $n_graphs \
                --num_basis_heads $num_basis_heads \
                --num_basis_layers $num_basis_layers \
                --diversifying_loss \
                --additional_FGW \
                --des MultiSource_VQVAE_Additional_FGW_diversifying_loss_20251106_heart_target_1
        done
    done
done

echo "=== 모든 실험 완료 ==="