gpu_id=5
source_dataset="adult"
few_shots="4 8 16 32 64"
model_types="GAT_edge_4"
FDs="N D"
center_types="CM CA_m CA_f"
random_seeds="42 1993 2025 1234 5678"
for random_seed in $random_seeds; do
    for shot in $few_shots; do
        for model_type in $model_types; do
            for FD in $FDs; do
                echo "Running experiment for $source_dataset with $shot few-shot samples in random_seed:$random_seed"
                CUDA_VISIBLE_DEVICES=$gpu_id python main_s2s.py \
                --random_seed $random_seed \
                --source_dataset_name $source_dataset \
                --base_dir 'optuna' \
                --few_shot $shot \
                --train_epochs 200 \
                --graph_type 'star' \
                --FD $FD \
                --center_type 'CM' \
                --model_type $model_type
                
            done
        done
    done
done