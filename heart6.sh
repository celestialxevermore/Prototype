#!/bin/bash
gpu_id=0
export CUDA_VISIBLE_DEVICES=$gpu_id
random_seeds="42"
few_shots="4 8 16 32 64"
num_basis_layers="2"
num_basis_heads="4"

# 시나리오 1: heart_target_1 heart_target_2 heart_target_3 heart_target_4 -> heart
# echo "=== 시나리오 1: Heart_disease_statlog Cardiovascular_Disease_Dataset heart_target_3 heart_target_4 -> heart_disease (diversifying_loss on/off) ==="

# for random_seed in $random_seeds; do
#     for few_shot in $few_shots; do
#         # Case 2: diversifying_loss ON
#         base_dir="test20251103_MultiSource_som_loss_NO_LCG_diversifying_loss_Target_heart_disease"
#         echo "Running experiment (div_loss=OFF) - seed:${random_seed}, few_shot:${few_shot}"
#         CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_S.py \
#             --random_seed $random_seed \
#             --source_data Heart_disease_statlog Cardiovascular_Disease_Dataset heart_target_3 heart_target_4 \
#             --target_data heart_disease \
#             --base_dir $base_dir \
#             --few_shot $few_shot \
#             --num_basis_heads $num_basis_heads \
#             --num_basis_layers $num_basis_layers \
#             --des MultiSource_NO_LCG_diversifying_loss_head_4_20251103

#     done
# done

echo "=== 시나리오 1: Heart_disease_statlog Cardiovascular_Disease_Dataset heart_target_3 heart_target_4 -> diabetes (diversifying_loss on/off) ==="


for random_seed in $random_seeds; do
    for few_shot in $few_shots; do
        # Case 2: diversifying_loss ON
        base_dir="test20251103_MultiSource_som_loss_NO_LCG_diversifying_loss_Target_diabete"
        echo "Running experiment (div_loss=OFF) - seed:${random_seed}, few_shot:${few_shot}"
        CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_S.py \
            --random_seed $random_seed \
            --source_data Heart_disease_statlog Cardiovascular_Disease_Dataset heart_target_3 heart_target_4 \
            --target_data diabetes \
            --base_dir $base_dir \
            --few_shot $few_shot \
            --num_basis_heads $num_basis_heads \
            --num_basis_layers $num_basis_layers \
            --des MultiSource_NO_LCG_diversifying_loss_head_4_20251103

    done
done




echo "=== 모든 실험 완료 ==="