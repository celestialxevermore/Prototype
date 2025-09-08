#!/bin/bash
gpu_id=4
export CUDA_VISIBLE_DEVICES=$gpu_id
random_seeds="42 44 46 48 50"
embed_types="carte ours"
edge_types="mlp"
attn_types="gat_v1"
few_shots="4 8 16 32 64"
num_basis_layers="2 3"
relation_scorer_types="pair_mlp query"


# 시나리오 1: heart_target_1 heart_target_2 heart_target_3 heart_target_4 -> heart
echo "=== 시나리오 1: heart_statlog heart1 Medicaldataset -> heart (gat_v1) ==="
for random_seed in $random_seeds; do
    for few_shot in $few_shots; do
        for embed_type in $embed_types; do
            for attn_type in $attn_types; do
                for edge_type in $edge_types; do
                    for num_basis in $num_basis_layers; do
                        for rel_scorer in $relation_scorer_types; do
                            base_dir="test20250902_new_gat_v1_embed_type_${embed_type}_num_basis_${num_basis}_mask_share_across_layers:False_scorer_${rel_scorer}_rel_symmetric:True_no_self_loop:False"
                            echo "Running experiment - heart_statlog heart1 Medicaldataset -> heart (gat_v1)"
                            CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_SS.py \
                            --random_seed $random_seed \
                            --source_data heart_statlog heart1 Medicaldataset \
                            --target_data heart \
                            --base_dir $base_dir \
                            --embed_type $embed_type \
                            --edge_type $edge_type \
                            --attn_type $attn_type \
                            --few_shot $few_shot \
                            --num_basis_layers $num_basis \
                            --relation_scorer_type $rel_scorer \
                            --rel_symmetric
                        done
                    done
                done
            done
        done
    done
done
