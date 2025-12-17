#!/bin/bash
gpu_id=4
#export CUDA_VISIBLE_DEVICES=$gpu_id
random_seeds="62 63 64 65 66"
few_shots="0 4 8 16 32 64"
num_basis_layers="2"
struct_dims="192 64"
taus="0.1 0.2 0.3"
echo "=== 시나리오 1 VQVAE: 'Heart_disease_statlog', 'Cardiovascular_Disease_Dataset', 'heart_target_3', 'heart_target_4' -> heart_target_2 ==="
for random_seed in $random_seeds; do
    for few_shot in $few_shots; do
        for struct_dim in $struct_dims; do
            for tau in $taus; do
                base_dir="test20251209_Global_v4_uniform_Case1_"
                echo "Running experiment - seed:${random_seed}, few_shot:${few_shot}"
                CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_S.py \
                    --random_seed $random_seed \
                    --source_data Heart_disease_statlog Cardiovascular_Disease_Dataset heart_target_3 heart_target_4 \
                    --target_data heart \
                    --base_dir $base_dir \
                    --few_shot $few_shot \
                    --struct_hidden_dim $struct_dim \
                    --tau $tau \
                    --num_basis_layers $num_basis_layers \
                    --des v4_Case1_uniform_
            done
        done
    done
done

# echo "=== 시나리오 1 VQVAE: 'Heart_disease_statlog', 'Cardiovascular_Disease_Dataset', 'heart_target_3', 'heart_target_4' -> heart_target_2 ==="
# for random_seed in $random_seeds; do

#     for few_shot in $few_shots; do
#         base_dir="test20251208_Global_v4_Case3_"
#         echo "Running experiment - seed:${random_seed}, few_shot:${few_shot}"
#         CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_S.py \
#             --random_seed $random_seed \
#             --source_data Heart_disease_statlog Cardiovascular_Disease_Dataset Medicaldataset heart_target_4 \
#             --target_data heart \
#             --base_dir $base_dir \
#             --few_shot $few_shot \
#             --num_basis_layers $num_basis_layers \
#             --des v4_Case3
#     done
# done



echo "=== 시나리오 1 VQVAE: 'Heart_disease_statlog Cardiovascular_Disease_Dataset Medicaldataset heart_target_4' -> heart ==="
for random_seed in $random_seeds; do
    for few_shot in $few_shots; do
        for struct_dim in $struct_dims; do
            for tau in $taus; do
                base_dir="test20251209_Global_v4_uniform_Case4_"
                echo "Running experiment - seed:${random_seed}, few_shot:${few_shot}"
                CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_S.py \
                    --random_seed $random_seed \
                    --source_data Heart_disease_statlog Cardiovascular_Disease_Dataset Medicaldataset heart_target_4 \
                    --target_data heart \
                    --base_dir $base_dir \
                    --few_shot $few_shot \
                    --strustruct_hidden_dimct_dim $struct_dim \
                    --tau $tau \
                    --num_basis_layers $num_basis_layers \
                    --des v4_Case4_uniform_
            done
        done
    done
done 

# echo "=== 시나리오 1 VQVAE: 'Cardiovascular_Disease_Dataset Medicaldataset heart heart_target_4' -> heart ==="
# for random_seed in $random_seeds; do
#     for few_shot in $few_shots; do
#         base_dir="test20251208_Global_v4_Case2_"
#         echo "Running experiment - seed:${random_seed}, few_shot:${few_shot}"
#         CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_S.py \
#             --random_seed $random_seed \
#             --source_data Cardiovascular_Disease_Dataset Medicaldataset heart heart_target_4 \
#             --target_data Heart_disease_statlog  \
#             --base_dir $base_dir \
#             --few_shot $few_shot \
#             --num_basis_layers $num_basis_layers \
#             --des v4_Case2
#     done
    
# done



echo "=== 모든 실험 완료 ==="