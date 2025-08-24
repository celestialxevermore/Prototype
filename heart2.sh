#!/bin/bash

#!/bin/bash
gpu_id=4
random_seeds="42 44 46 48 50"
embed_types="carte"
edge_types="mlp"
attn_types="gat_v1"
few_shots="4 8 16 32 64"




# 시나리오 1: heart_target_1,2,3 -> heart
echo "=== 시나리오 1: heart_target_1,2,3 -> heart ==="
for random_seed in $random_seeds; do
    for few_shot in $few_shots; do
        for embed_type in $embed_types; do
            for attn_type in $attn_types; do
                for edge_type in $edge_types; do
                    echo "Running experiment - diabetes credit-g breast -> heart"
                    CUDA_VISIBLE_DEVICES=$gpu_id python main_S.py \
                    --random_seed $random_seed \
                    --source_data diabetes credit-g breast \
                    --target_data heart \
                    --base_dir 'test20250825' \
                    --embed_type $embed_type \
                    --edge_type $edge_type \
                    --attn_type $attn_type \

                done
            done
        done
    done
done

# 시나리오 1: heart_target_1,2,3 -> heart
echo "=== 시나리오 1: heart_target_1,2,3 -> heart ==="
for random_seed in $random_seeds; do
    for few_shot in $few_shots; do
        for embed_type in $embed_types; do
            for attn_type in $attn_types; do
                for edge_type in $edge_types; do
                    echo "Running experiment - diabetes credit-g breast heart_target_1 -> heart"
                    CUDA_VISIBLE_DEVICES=$gpu_id python main_S.py \
                    --random_seed $random_seed \
                    --source_data diabetes credit-g breast heart_target_1 \
                    --target_data heart \
                    --base_dir 'test20250825' \
                    --embed_type $embed_type \
                    --edge_type $edge_type \
                    --attn_type $attn_type \

                done
            done
        done
    done
done

# 시나리오 2: diabetes,credit-g,blood,heart_target_1 -> heart
echo "=== 시나리오 2: diabetes,credit-g,blood,heart_target_1 -> heart ==="
for random_seed in $random_seeds; do
    for few_shot in $few_shots; do
        for embed_type in $embed_types; do
            for attn_type in $attn_types; do
                for edge_type in $edge_types; do
                    echo "Running experiment - breast credit-g heart -> diabetes"
                    CUDA_VISIBLE_DEVICES=$gpu_id python main_S.py \
                    --random_seed $random_seed \
                    --source_data breast credit-g heart \
                    --target_data diabetes \
                    --base_dir 'test20250825' \
                    --embed_type $embed_type \
                    --edge_type $edge_type \
                    --attn_type $attn_type \
                    --few_shot $few_shot \

                done
            done
        done
    done
done

# 시나리오 2: diabetes,credit-g,blood,heart_target_1 -> heart
echo "=== 시나리오 2: diabetes,credit-g,blood,heart_target_1 -> heart ==="
for random_seed in $random_seeds; do
    for few_shot in $few_shots; do
        for embed_type in $embed_types; do
            for attn_type in $attn_types; do
                for edge_type in $edge_types; do
                    echo "Running experiment - breast credit-g heart heart_target_1 -> diabetes"
                    CUDA_VISIBLE_DEVICES=$gpu_id python main_S.py \
                    --random_seed $random_seed \
                    --source_data breast credit-g heart heart_target_1 \
                    --target_data diabetes \
                    --base_dir 'test20250825' \
                    --embed_type $embed_type \
                    --edge_type $edge_type \
                    --attn_type $attn_type \
                    --few_shot $few_shot \

                done
            done
        done
    done
done


# for random_seed in $random_seeds; do
#    for few_shot in $few_shots; do
#        for embed_type in $embed_types; do
#            for attn_type in $attn_types; do
#                for edge_type in $edge_types; do 
#                    for source_dataset in $source_datasets; do
                       
#                        # Self-loop 유지 버전
#                        echo "Running experiment - WITH self-loop"
#                        CUDA_VISIBLE_DEVICES=$gpu_id python main_PP.py \
#                        --random_seed $random_seed \
#                        --source_data $source_dataset \
#                        --base_dir 'test20250726_with_self_loop_prototypePPPP' \
#                        --embed_type $embed_type \
#                        --edge_type $edge_type \
#                        --attn_type $attn_type \
#                        --few_shot $few_shot \
#                        --train_epochs 1000 \
#                        --model_type "TabularFLM"

#                        # Self-loop 제거 버전  
#                        echo "Running experiment - WITHOUT self-loop"
#                        CUDA_VISIBLE_DEVICES=$gpu_id python main_PP.py \
#                        --random_seed $random_seed \
#                        --source_data $source_dataset \
#                        --base_dir 'test20250726_no_self_loop_prototypePPPP' \
#                        --embed_type $embed_type \
#                        --edge_type $edge_type \
#                        --attn_type $attn_type \
#                        --few_shot $few_shot \
#                        --train_epochs 1000 \
#                        --model_type "TabularFLM" \
#                        --no_self_loop
                       
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