#!/bin/bash

# GPU ID 설정
gpu_id=0

# OpenMP 스레드 수 설정
export OMP_NUM_THREADS=10

# 선택 가능한 값들
llm_models="gpt2_mean gpt2_auto sentence-bert bio-bert bio-clinical-bert LLAMA_mean LLAMA_auto"
embed_types="ours2"

# 실행 루프
for llm_model in $llm_models; do
    for embed_type in $embed_types; do
        echo "▶ Running make_embed_dataset.py with llm_model=$llm_model, embed_type=$embed_type"

        CUDA_VISIBLE_DEVICES=$gpu_id python make_embed_dataset.py \
            --llm_model "$llm_model" \
            --embed_type "$embed_type"

        echo "✅ Completed: llm_model=$llm_model, embed_type=$embed_type"
        echo ""
    done
done
