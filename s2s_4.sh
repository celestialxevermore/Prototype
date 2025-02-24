gpu_id=4
source_datasets="adult heart diabetes"
few_shots="4 8 16 32 64"
model_types="TabularFLM"
labels="add no"
modes="sa mean"
for source_dataset in $source_datasets; do
    for shot in $few_shots; do
        for label in $labels; do
            for mode in $modes; do 
                echo "Running experiment for $source_dataset with $shot few-shot samples in random_seed:$random_seed"
                CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
                --random_seed 1234 \
                --source_dataset_name $source_dataset \
                --base_dir 'Experiment5' \
                --few_shot $shot \
                --train_epochs 200 \
                --label $label \
                --mode $mode \
                --model_type "TabularFLM"
            done
        done
    done
done
