#!/bin/bash

# Set OMP threads
export OMP_NUM_THREADS=10

# Run analysis1.py for all attention map directories
echo "Running analysis1.py for all attention map directories..."

python analysis1.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-carte_desc_Edge-False_A-att"
echo "Completed: carte_desc_Edge-False_A-att"

python analysis1.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-carte_desc_Edge-False_A-gat"
echo "Completed: carte_desc_Edge-False_A-gat"

python analysis1.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-carte_desc_Edge-True_A-att"
echo "Completed: carte_desc_Edge-True_A-att"

python analysis1.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-carte_desc_Edge-True_A-gat"
echo "Completed: carte_desc_Edge-True_A-gat"

python analysis1.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-carte_Edge-False_A-att"
echo "Completed: carte_Edge-False_A-att"

python analysis1.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-carte_Edge-False_A-gat"
echo "Completed: carte_Edge-False_A-gat"

python analysis1.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-carte_Edge-True_A-att"
echo "Completed: carte_Edge-True_A-att"

python analysis1.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-carte_Edge-True_A-gat"
echo "Completed: carte_Edge-True_A-gat"

python analysis1.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-ours_Edge-False_A-att"
echo "Completed: ours_Edge-False_A-att"

python analysis1.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-ours_Edge-False_A-gat"
echo "Completed: ours_Edge-False_A-gat"

python analysis1.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-ours_Edge-True_A-att"
echo "Completed: ours_Edge-True_A-att"

python analysis1.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-ours_Edge-True_A-gat"
echo "Completed: ours_Edge-True_A-gat"

python analysis1.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-ours2_Edge-False_A-att"
echo "Completed: ours2_Edge-False_A-att"

python analysis1.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-ours2_Edge-False_A-gat"
echo "Completed: ours2_Edge-False_A-gat"

python analysis1.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-ours2_Edge-True_A-att"
echo "Completed: ours2_Edge-True_A-att"

python analysis1.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-ours2_Edge-True_A-gat"
echo "Completed: ours2_Edge-True_A-gat"

echo "All analyses completed!"