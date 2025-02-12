gpu_id=2
source_datasets="heart diabetes"
few_shots="4 8 16 32 64"
batch_sizes="32 64"
FDs="N D"

for source_dataset in $source_datasets; do
    for shot in $few_shots; do
        for batch_size in $batch_sizes; do
            for FD in $FDs; do
                echo "Running experiment for $source_dataset with $shot few-shot samples"
                CUDA_VISIBLE_DEVICES=$gpu_id python main_s2s.py \
                    --source_dataset_name $source_dataset \
                    --few_shot $shot \
                    --batch_size $batch_size \
                    --num_layers 3 \
                    --heads 4 \
                    --dropout_rate 0.2 \
                    --FD $FD \
                    --center_type 'CM' \
                    --model_type 'GAT_edge_2'
            done
        done
    done
done