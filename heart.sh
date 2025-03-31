gpu_id=4
source_datasets="heart"
few_shots="4 8 16 32 64"
model_types="TabularFLM"
#random_seeds="2095 3192 3155 67 1045 "
# 2000부터 2199까지 200개의 랜덤 시드로 반복



for few_shot in $few_shots; do 
    for source_dataset in $source_datasets; do
        echo "Running experiment for $source_dataset with $few_shot few-shot samples in random_seed:$random_seed"
        CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_G4.py \
        --random_seed 2095 \
        --source_dataset_name $source_dataset \
        --base_dir 'tabular_embedding_new_gpt2_mean' \
        --llm_model 'gpt2_mean' \
        --input_dim 768 \
        --use_gmm \
        --few_shot $few_shot \
        --train_epochs 1000 \
        --model_type "TabularFLM"
    done
done

for few_shot in $few_shots; do 
    for source_dataset in $source_datasets; do
        echo "Running experiment for $source_dataset with $few_shot few-shot samples in random_seed:$random_seed"
        CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_G4.py \
        --random_seed 2095 \
        --source_dataset_name $source_dataset \
        --base_dir 'tabular_embedding_new_gpt2_auto' \
        --llm_model 'gpt2_auto' \
        --input_dim 768 \
        --use_gmm \
        --few_shot $few_shot \
        --train_epochs 1000 \
        --model_type "TabularFLM"
    done
done

for few_shot in $few_shots; do 
    for source_dataset in $source_datasets; do
        echo "Running experiment for $source_dataset with $few_shot few-shot samples in random_seed:$random_seed"
        CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_G4.py \
        --random_seed 2095 \
        --source_dataset_name $source_dataset \
        --base_dir 'tabular_embedding_new_sentence-bert' \
        --llm_model 'sentence-bert' \
        --input_dim 384 \
        --use_gmm \
        --few_shot $few_shot \
        --train_epochs 1000 \
        --model_type "TabularFLM"
    done
done

for few_shot in $few_shots; do 
    for source_dataset in $source_datasets; do
        echo "Running experiment for $source_dataset with $few_shot few-shot samples in random_seed:$random_seed"
        CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_G4.py \
        --random_seed 2095 \
        --source_dataset_name $source_dataset \
        --base_dir 'tabular_embedding_new_bio-bert' \
        --llm_model 'bio-bert' \
        --input_dim 768 \
        --use_gmm \
        --few_shot $few_shot \
        --train_epochs 1000 \
        --model_type "TabularFLM"
    done
done

for few_shot in $few_shots; do 
    for source_dataset in $source_datasets; do
        echo "Running experiment for $source_dataset with $few_shot few-shot samples in random_seed:$random_seed"
        CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_G4.py \
        --random_seed 2095 \
        --source_dataset_name $source_dataset \
        --base_dir 'tabular_embedding_old_bio-clinical-bert' \
        --llm_model 'bio-clinical-bert' \
        --input_dim 768 \
        --use_gmm \
        --few_shot $few_shot \
        --train_epochs 1000 \
        --model_type "TabularFLM"
    done
done

for few_shot in $few_shots; do 
    for source_dataset in $source_datasets; do
        echo "Running experiment for $source_dataset with $few_shot few-shot samples in random_seed:$random_seed"
        CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_G4.py \
        --random_seed 2095 \
        --source_dataset_name $source_dataset \
        --base_dir 'tabular_embedding_new_LLAMA_mean' \
        --llm_model 'LLAMA_mean' \
        --input_dim 4096 \
        --use_gmm \
        --few_shot $few_shot \
        --train_epochs 1000 \
        --model_type "TabularFLM"
    done
done

for few_shot in $few_shots; do 
    for source_dataset in $source_datasets; do
        echo "Running experiment for $source_dataset with $few_shot few-shot samples in random_seed:$random_seed"
        CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_G4.py \
        --random_seed 2095 \
        --source_dataset_name $source_dataset \
        --base_dir 'tabular_embedding_new_LLAMA_auto' \
        --llm_model 'LLAMA_auto' \
        --input_dim 4096 \
        --use_gmm \
        --few_shot $few_shot \
        --train_epochs 1000 \
        --model_type "TabularFLM"
    done
done