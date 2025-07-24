#!/bin/bash

# 색상 에러 방지
export WANDB_CONSOLE=off
export NO_COLOR=1

gpu_id=0
source_datasets="heart"
random_seeds="42 44 46 48 50"
embed_types="carte"
edge_types="mlp"
attn_types="gat_v2"  # gat_v1과 gat_v2 따로 실행
few_shots="4 8 16 32 64"
learning_rates="0.001 0.0001 0.00001"
learning_rates_few="0.001 0.0001 0.00001"
hidden_dims="256 512 768"
num_layers_list="2 3 4"
n_heads_list="2 4 8"
dropout_rates="0.1 0.2 0.3"

for lr in $learning_rates; do
    for lr_few in $learning_rates_few; do
        for hidden in $hidden_dims; do
            for layers in $num_layers_list; do
                for heads in $n_heads_list; do
                    for dropout in $dropout_rates; do
                        for random_seed in $random_seeds; do
                           for few_shot in $few_shots; do
                               for embed_type in $embed_types; do
                                   for attn_type in $attn_types; do
                                       for edge_type in $edge_types; do 
                                           for source_dataset in $source_datasets; do
                                               
                                               echo "Running experiment - LR: $lr, LR_few: $lr_few, Hidden: $hidden, Layers: $layers, Heads: $heads, Dropout: $dropout, Seed: $random_seed, Attn: $attn_type"
                                               CUDA_VISIBLE_DEVICES=$gpu_id python main_G.py \
                                               --random_seed $random_seed \
                                               --source_data $source_dataset \
                                               --base_dir 'wandbtest20250722' \
                                               --embed_type $embed_type \
                                               --edge_type $edge_type \
                                               --attn_type $attn_type \
                                               --few_shot $few_shot \
                                               --train_epochs 1000 \
                                               --model_type "TabularFLM" \
                                               --source_lr $lr \
                                               --source_lr_few $lr_few \
                                               --hidden_dim $hidden \
                                               --num_layers $layers \
                                               --n_heads $heads \
                                               --dropout_rate $dropout \
                                               --use_wandb \
                                               --wandb_project "tabular-hyperparams-direct-gat_v2"
                                            
                                           done
                                       done
                                   done
                               done
                           done
                        done
                    done
                done
            done
        done
    done
done