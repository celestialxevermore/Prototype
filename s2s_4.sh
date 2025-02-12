
gpu_id=3
source_dataset="heart diabetes adult"
few_shots="4 8 16 32 64"
batch_sizes="32 64"
model_types="GAT_edge_2 GAT_edge_3"
FDs="N D ND"
center_types="CM"
random_seeds="1234"
for random_seed in $random_seeds; do
    for shot in $few_shots; do
        for model_type in $model_types; do
            for batch_size in $batch_sizes; do
                for FD in $FDs; do
                    echo "Running experiment for $source_dataset with $shot few-shot samples in random_seed:$random_seed"
                    CUDA_VISIBLE_DEVICES=$gpu_id python main_s2s_few.py \
                    --random_seed $random_seed \
                    --source_dataset_name $source_dataset \
                    --few_shot $shot \
                    --train_epochs 200 \
                    --num_layers 3 \
                    --heads 4 \
                    --dropout_rate 0.2 \
                    --batch_size $batch_size \
                    --graph_type 'star' \
                    --FD $FD \
                    --center_type 'CM' \
                    --model_type $model_type
                    
                done
            done
        done
    done
done

for random_seed in $random_seeds; do
    for shot in $few_shots; do
        for model_type in $model_types; do
            for batch_size in $batch_sizes; do
                for FD in $FDs; do
                    echo "Running experiment for $source_dataset with $shot few-shot samples in random_seed:$random_seed"
                    CUDA_VISIBLE_DEVICES=$gpu_id python main_s2s_few.py \
                    --random_seed $random_seed \
                    --source_dataset_name $source_dataset \
                    --few_shot $shot \
                    --train_epochs 200 \
                    --batch_size $batch_size \
                    --graph_type 'star' \
                    --FD $FD \
                    --center_type 'CM' \
                    --model_type $model_type
                    
                done
            done
        done
    done
done