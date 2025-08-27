#!/bin/bash
gpu_id=4
random_seeds="42 44 46 48 50"
embed_types="carte"
edge_types="mlp"
attn_types="gat_v1"
few_shots="4 8 16 32 64"

gpu_id=4
random_seeds="42 44 46 48 50"
embed_types="carte"
edge_types="mlp"
attn_types="gat_v1"
few_shots="4 8 16 32 64"

# 시나리오 1: diabetes,credit-g,breast -> heart
echo "=== 시나리오 1: diabetes,credit-g,breast -> heart ==="
for random_seed in $random_seeds; do
    for few_shot in $few_shots; do
        for embed_type in $embed_types; do
            for attn_type in $attn_types; do
                for edge_type in $edge_types; do
                    echo "Running experiment - diabetes,credit-g,breast -> heart (S_baseline_B_baseline)"
                    CUDA_VISIBLE_DEVICES=$gpu_id python main_S.py \
                    --random_seed $random_seed \
                    --source_data diabetes credit-g breast \
                    --target_data heart \
                    --base_dir 'test20250825/S_baseline_B_baseline' \
                    --embed_type $embed_type \
                    --edge_type $edge_type \
                    --attn_type $attn_type \
                    --few_shot $few_shot \
                    --shared_connectivity baseline \
                    --basis_connectivity baseline

                    echo "Running experiment - diabetes,credit-g,breast -> heart (S_cls2var_B_baseline)"
                    CUDA_VISIBLE_DEVICES=$gpu_id python main_S.py \
                    --random_seed $random_seed \
                    --source_data diabetes credit-g breast \
                    --target_data heart \
                    --base_dir 'test20250825/S_cls2var_B_baseline' \
                    --embed_type $embed_type \
                    --edge_type $edge_type \
                    --attn_type $attn_type \
                    --few_shot $few_shot \
                    --shared_connectivity cls2var \
                    --basis_connectivity baseline

                    echo "Running experiment - diabetes,credit-g,breast -> heart (S_baseline_B_full)"
                    CUDA_VISIBLE_DEVICES=$gpu_id python main_S.py \
                    --random_seed $random_seed \
                    --source_data diabetes credit-g breast \
                    --target_data heart \
                    --base_dir 'test20250825/S_baseline_B_full' \
                    --embed_type $embed_type \
                    --edge_type $edge_type \
                    --attn_type $attn_type \
                    --few_shot $few_shot \
                    --shared_connectivity baseline \
                    --basis_connectivity full

                    echo "Running experiment - diabetes,credit-g,breast -> heart (S_cls2var_B_full)"
                    CUDA_VISIBLE_DEVICES=$gpu_id python main_S.py \
                    --random_seed $random_seed \
                    --source_data diabetes credit-g breast \
                    --target_data heart \
                    --base_dir 'test20250825/S_cls2var_B_full' \
                    --embed_type $embed_type \
                    --edge_type $edge_type \
                    --attn_type $attn_type \
                    --few_shot $few_shot \
                    --shared_connectivity cls2var \
                    --basis_connectivity full
                done
            done
        done
    done
done

# 시나리오 2: diabetes,credit-g,breast,heart_target_1 -> heart
echo "=== 시나리오 2: diabetes,credit-g,breast,heart_target_1 -> heart ==="
for random_seed in $random_seeds; do
    for few_shot in $few_shots; do
        for embed_type in $embed_types; do
            for attn_type in $attn_types; do
                for edge_type in $edge_types; do
                    echo "Running experiment - diabetes,credit-g,breast,heart_target_1 -> heart (S_baseline_B_baseline)"
                    CUDA_VISIBLE_DEVICES=$gpu_id python main_S.py \
                    --random_seed $random_seed \
                    --source_data diabetes credit-g breast heart_target_1 \
                    --target_data heart \
                    --base_dir 'test20250825/S_baseline_B_baseline' \
                    --embed_type $embed_type \
                    --edge_type $edge_type \
                    --attn_type $attn_type \
                    --few_shot $few_shot \
                    --shared_connectivity baseline \
                    --basis_connectivity baseline

                    echo "Running experiment - diabetes,credit-g,breast,heart_target_1 -> heart (S_cls2var_B_baseline)"
                    CUDA_VISIBLE_DEVICES=$gpu_id python main_S.py \
                    --random_seed $random_seed \
                    --source_data diabetes credit-g breast heart_target_1 \
                    --target_data heart \
                    --base_dir 'test20250825/S_cls2var_B_baseline' \
                    --embed_type $embed_type \
                    --edge_type $edge_type \
                    --attn_type $attn_type \
                    --few_shot $few_shot \
                    --shared_connectivity cls2var \
                    --basis_connectivity baseline

                    echo "Running experiment - diabetes,credit-g,breast,heart_target_1 -> heart (S_baseline_B_full)"
                    CUDA_VISIBLE_DEVICES=$gpu_id python main_S.py \
                    --random_seed $random_seed \
                    --source_data diabetes credit-g breast heart_target_1 \
                    --target_data heart \
                    --base_dir 'test20250825/S_baseline_B_full' \
                    --embed_type $embed_type \
                    --edge_type $edge_type \
                    --attn_type $attn_type \
                    --few_shot $few_shot \
                    --shared_connectivity baseline \
                    --basis_connectivity full

                    echo "Running experiment - diabetes,credit-g,breast,heart_target_1 -> heart (S_cls2var_B_full)"
                    CUDA_VISIBLE_DEVICES=$gpu_id python main_S.py \
                    --random_seed $random_seed \
                    --source_data diabetes credit-g breast heart_target_1 \
                    --target_data heart \
                    --base_dir 'test20250825/S_cls2var_B_full' \
                    --embed_type $embed_type \
                    --edge_type $edge_type \
                    --attn_type $attn_type \
                    --few_shot $few_shot \
                    --shared_connectivity cls2var \
                    --basis_connectivity full
                done
            done
        done
    done
done

# 시나리오 3: breast,credit-g,heart -> diabetes
echo "=== 시나리오 3: breast,credit-g,heart -> diabetes ==="
for random_seed in $random_seeds; do
    for few_shot in $few_shots; do
        for embed_type in $embed_types; do
            for attn_type in $attn_types; do
                for edge_type in $edge_types; do
                    echo "Running experiment - breast,credit-g,heart -> diabetes (S_baseline_B_baseline)"
                    CUDA_VISIBLE_DEVICES=$gpu_id python main_S.py \
                    --random_seed $random_seed \
                    --source_data breast credit-g heart \
                    --target_data diabetes \
                    --base_dir 'test20250825/S_baseline_B_baseline' \
                    --embed_type $embed_type \
                    --edge_type $edge_type \
                    --attn_type $attn_type \
                    --few_shot $few_shot \
                    --shared_connectivity baseline \
                    --basis_connectivity baseline

                    echo "Running experiment - breast,credit-g,heart -> diabetes (S_cls2var_B_baseline)"
                    CUDA_VISIBLE_DEVICES=$gpu_id python main_S.py \
                    --random_seed $random_seed \
                    --source_data breast credit-g heart \
                    --target_data diabetes \
                    --base_dir 'test20250825/S_cls2var_B_baseline' \
                    --embed_type $embed_type \
                    --edge_type $edge_type \
                    --attn_type $attn_type \
                    --few_shot $few_shot \
                    --shared_connectivity cls2var \
                    --basis_connectivity baseline

                    echo "Running experiment - breast,credit-g,heart -> diabetes (S_baseline_B_full)"
                    CUDA_VISIBLE_DEVICES=$gpu_id python main_S.py \
                    --random_seed $random_seed \
                    --source_data breast credit-g heart \
                    --target_data diabetes \
                    --base_dir 'test20250825/S_baseline_B_full' \
                    --embed_type $embed_type \
                    --edge_type $edge_type \
                    --attn_type $attn_type \
                    --few_shot $few_shot \
                    --shared_connectivity baseline \
                    --basis_connectivity full

                    echo "Running experiment - breast,credit-g,heart -> diabetes (S_cls2var_B_full)"
                    CUDA_VISIBLE_DEVICES=$gpu_id python main_S.py \
                    --random_seed $random_seed \
                    --source_data breast credit-g heart \
                    --target_data diabetes \
                    --base_dir 'test20250825/S_cls2var_B_full' \
                    --embed_type $embed_type \
                    --edge_type $edge_type \
                    --attn_type $attn_type \
                    --few_shot $few_shot \
                    --shared_connectivity cls2var \
                    --basis_connectivity full
                done
            done
        done
    done
done

# 시나리오 4: breast,credit-g,heart,heart_target_1 -> diabetes
echo "=== 시나리오 4: breast,credit-g,heart,heart_target_1 -> diabetes ==="
for random_seed in $random_seeds; do
    for few_shot in $few_shots; do
        for embed_type in $embed_types; do
            for attn_type in $attn_types; do
                for edge_type in $edge_types; do
                    echo "Running experiment - breast,credit-g,heart,heart_target_1 -> diabetes (S_baseline_B_baseline)"
                    CUDA_VISIBLE_DEVICES=$gpu_id python main_S.py \
                    --random_seed $random_seed \
                    --source_data breast credit-g heart heart_target_1 \
                    --target_data diabetes \
                    --base_dir 'test20250825/S_baseline_B_baseline' \
                    --embed_type $embed_type \
                    --edge_type $edge_type \
                    --attn_type $attn_type \
                    --few_shot $few_shot \
                    --shared_connectivity baseline \
                    --basis_connectivity baseline

                    echo "Running experiment - breast,credit-g,heart,heart_target_1 -> diabetes (S_cls2var_B_baseline)"
                    CUDA_VISIBLE_DEVICES=$gpu_id python main_S.py \
                    --random_seed $random_seed \
                    --source_data breast credit-g heart heart_target_1 \
                    --target_data diabetes \
                    --base_dir 'test20250825/S_cls2var_B_baseline' \
                    --embed_type $embed_type \
                    --edge_type $edge_type \
                    --attn_type $attn_type \
                    --few_shot $few_shot \
                    --shared_connectivity cls2var \
                    --basis_connectivity baseline

                    echo "Running experiment - breast,credit-g,heart,heart_target_1 -> diabetes (S_baseline_B_full)"
                    CUDA_VISIBLE_DEVICES=$gpu_id python main_S.py \
                    --random_seed $random_seed \
                    --source_data breast credit-g heart heart_target_1 \
                    --target_data diabetes \
                    --base_dir 'test20250825/S_baseline_B_full' \
                    --embed_type $embed_type \
                    --edge_type $edge_type \
                    --attn_type $attn_type \
                    --few_shot $few_shot \
                    --shared_connectivity baseline \
                    --basis_connectivity full

                    echo "Running experiment - breast,credit-g,heart,heart_target_1 -> diabetes (S_cls2var_B_full)"
                    CUDA_VISIBLE_DEVICES=$gpu_id python main_S.py \
                    --random_seed $random_seed \
                    --source_data breast credit-g heart heart_target_1 \
                    --target_data diabetes \
                    --base_dir 'test20250825/S_cls2var_B_full' \
                    --embed_type $embed_type \
                    --edge_type $edge_type \
                    --attn_type $attn_type \
                    --few_shot $few_shot \
                    --shared_connectivity cls2var \
                    --basis_connectivity full
                done
            done
        done
    done
done



# Heart에서 절대 제거하면 안 되는 변수들
# ST_Slope Oldpeak Age

# 제거 가능한 변수들 (random하게 선택)
# del_feat_configs=(
#     "Sex ChestPainType"
#     "RestingECG ExerciseAngina"
#     "RestingBP Cholesterol"
#     "FastingBS MaxHR"
#     "Sex RestingECG"
#     "ChestPainType ExerciseAngina"
#     "RestingBP FastingBS"
#     "Cholesterol MaxHR"
#     "Sex ChestPainType RestingECG"
#     "RestingECG ExerciseAngina RestingBP"
#     "Cholesterol FastingBS MaxHR"
#     "Sex RestingECG RestingBP"
#     "ChestPainType ExerciseAngina Cholesterol"
#     "RestingBP FastingBS MaxHR"
#     "Sex ChestPainType RestingECG ExerciseAngina"
#     "RestingECG RestingBP Cholesterol"
#     "ExerciseAngina Cholesterol FastingBS"
#     "RestingBP FastingBS MaxHR"
#     "Sex ChestPainType RestingECG ExerciseAngina RestingBP"
#     "ChestPainType RestingECG ExerciseAngina Cholesterol"
#     "RestingECG ExerciseAngina RestingBP FastingBS"
#     "ExerciseAngina RestingBP Cholesterol FastingBS"
#     "RestingECG RestingBP Cholesterol FastingBS"
#     "Sex ChestPainType RestingECG ExerciseAngina RestingBP Cholesterol"
#     "ChestPainType RestingECG ExerciseAngina RestingBP FastingBS"
#     "RestingECG ExerciseAngina RestingBP Cholesterol FastingBS"
#     "Sex ChestPainType RestingECG ExerciseAngina RestingBP Cholesterol FastingBS"
#     "ChestPainType RestingECG ExerciseAngina RestingBP Cholesterol FastingBS MaxHR"
#     "Sex ChestPainType RestingECG ExerciseAngina RestingBP Cholesterol FastingBS MaxHR"
# )

# for random_seed in $random_seeds; do
#    for few_shot in $few_shots; do
#        for embed_type in $embed_types; do
#            for attn_type in $attn_types; do
#                for edge_type in $edge_types; do 
#                    for source_dataset in $source_datasets; do
#                        for del_feat in "${del_feat_configs[@]}"; do
                           
#                            # Self-loop 유지 버전
#                            echo "Running experiment - WITH self-loop, del_feat: $del_feat"
#                            CUDA_VISIBLE_DEVICES=$gpu_id python main_G.py \
#                            --random_seed $random_seed \
#                            --source_data $source_dataset \
#                            --base_dir 'test20250812_perturbation_self_loop' \
#                            --embed_type $embed_type \
#                            --edge_type $edge_type \
#                            --attn_type $attn_type \
#                            --few_shot $few_shot \
#                            --train_epochs 1000 \
#                            --model_type "TabularFLM" \
#                            --del_feat $del_feat

#                            # Self-loop 제거 버전  
#                            echo "Running experiment - WITHOUT self-loop, del_feat: $del_feat"
#                            CUDA_VISIBLE_DEVICES=$gpu_id python main_G.py \
#                            --random_seed $random_seed \
#                            --source_data $source_dataset \
#                            --base_dir 'test20250812_perturbation_no_self_loop' \
#                            --embed_type $embed_type \
#                            --edge_type $edge_type \
#                            --attn_type $attn_type \
#                            --few_shot $few_shot \
#                            --train_epochs 1000 \
#                            --model_type "TabularFLM" \
#                            --no_self_loop \
#                            --del_feat $del_feat
                           
#                        done
#                    done
#                done
#            done
#        done
#    done
# done


# gpu_id=4
# source_datasets="heart"
# random_seeds="2095 3192 3155 67 1045"
# embed_types="carte"
# edge_types="mlp"
# attn_types="gat_v1"
# few_shots="4 8 16 32 64"

# for random_seed in $random_seeds; do
#   for few_shot in $few_shots; do
#       for embed_type in $embed_types; do
#           for attn_type in $attn_types; do
#               for edge_type in $edge_types; do 
#                   for source_dataset in $source_datasets; do
                      
#                       # Self-loop 유지 버전
#                       echo "Running experiment - WITH self-loop"
#                       CUDA_VISIBLE_DEVICES=$gpu_id python main_A.py \
#                       --random_seed $random_seed \
#                       --source_data $source_dataset \
#                       --base_dir 'test20250726_scenario12345_with_self_loop_H_D_L_2' \
#                       --embed_type $embed_type \
#                       --edge_type $edge_type \
#                       --attn_type $attn_type \
#                       --dropout_rate 0.5 \
#                       --n_heads 8 \
#                       --num_layers 2 \
#                       --few_shot $few_shot \
#                       --train_epochs 1000 \
#                       --model_type "TabularFLM"

#                       # Self-loop 제거 버전  
#                       echo "Running experiment - WITHOUT self-loop"
#                       CUDA_VISIBLE_DEVICES=$gpu_id python main_A.py \
#                       --random_seed $random_seed \
#                       --source_data $source_dataset \
#                       --base_dir 'test20250726_scenario12345_no_self_loop_H_D_L_2' \
#                       --embed_type $embed_type \
#                       --edge_type $edge_type \
#                       --attn_type $attn_type \
#                       --dropout_rate 0.5 \
#                       --n_heads 8 \
#                       --num_layers 2 \
#                       --few_shot $few_shot \
#                       --train_epochs 1000 \
#                       --model_type "TabularFLM" \
#                       --no_self_loop
                      
#                   done
#               done
#           done
#       done
#   done
# done