#!/bin/bash

gpu_id=1
source_datasets="diabetes"
seeds="42 1993 2025 1234 5678"
labels="add no"
enc_types="ind shared"
meta_types="meta_attn meta_mlp"
aggr_types="flatten mean attn"

for source_dataset in $source_datasets; do
    for seed in $seeds; do
        for label in $labels; do
            for enc_type in $enc_types; do
                for meta_type in $meta_types; do
                    for aggr_type in $aggr_types; do
                        echo "Running optimization for $source_dataset with seed:$seed, label:$label, enc:$enc_type, meta:$meta_type, aggr:$aggr_type"
                        CUDA_VISIBLE_DEVICES=$gpu_id python main_opt.py \
                        --random_seed $seed \
                        --source_dataset_name $source_dataset \
                        --base_dir 'Experiment_TabularFLM2' \
                        --label $label \
                        --enc_type $enc_type \
                        --meta_type $meta_type \
                        --aggr_type $aggr_type \
                        --n_trials 20
                    done
                done
            done
        done
    done
done