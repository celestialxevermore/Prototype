#!/bin/bash

gpu_id=4
source_datasets="adult"
random_seeds="2021"
embed_types="ours"
edge_types="mlp normal no_use"
attn_types="gat att"
few_shots="4 8 16 32 64"

for few_shot in $few_shots; do
   for embed_type in $embed_types; do
       for attn_type in $attn_types; do
           for edge_type in $edge_types; do 
               for source_dataset in $source_datasets; do
                   echo "Running experiment for $source_dataset with $few_shot few-shot samples, embed_type:$embed_type, edge_type:$edge_type, attn_type:$attn_type, random_seed:$random_seed"
                   
                   CUDA_VISIBLE_DEVICES=$gpu_id python main_G.py \
                   --random_seed 2021 \
                   --source_data $source_dataset \
                   --base_dir 'test20250618' \
                   --embed_type $embed_type \
                   --edge_type $edge_type \
                   --attn_type $attn_type \
                   --few_shot $few_shot \
                   --train_epochs 1000 \
                   --model_type "TabularFLM"
                done
            done
        done
    done
done