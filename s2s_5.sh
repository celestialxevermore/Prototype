gpu_id=5
source_datasets="heart diabetes adult"
few_shots="4 8 16 32 64"
model_types="TabularFLM"
aggr_types="flatten mean attn"
enc_types="ind shared"
meta_types="meta_mlp meta_attn"
labels="add no"

for few_shot in $few_shots; do
    for source_dataset in $source_datasets; do
        for agg_type in $aggr_types; do 
            for enc_type in $enc_types; do
                for meta_type in $meta_types; do
                    for label in $labels; do
                            echo "Running experiment for $source_dataset with $shot few-shot samples in random_seed:5678"
                            CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
                            --random_seed 5678 \
                            --source_dataset_name $source_dataset \
                            --base_dir 'Experiment_TabularFLM2' \
                            --few_shot $few_shot \
                            --train_epochs 1000 \
                            --label $label \
                            --aggr_type $agg_type \
                            --enc_type $enc_type \
                            --meta_type $meta_type \
                            --model_type "TabularFLM"
                    done
                done
            done
        done
    done
done