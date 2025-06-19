#!/bin/bash

# Set OMP threads
export OMP_NUM_THREADS=10

# Run clustering2.py for each checkpoint
echo "Running clustering2.py for all checkpoints..."

python clustering2.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-carte_desc_Edge-mlp_A-att"
echo "Completed: Embed-carte_desc_Edge-mlp_A-att"

python clustering2.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-carte_desc_Edge-mlp_A-gat"
echo "Completed: Embed-carte_desc_Edge-mlp_A-gat"

python clustering2.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-carte_desc_Edge-no_use_A-att"
echo "Completed: Embed-carte_desc_Edge-no_use_A-att"

python clustering2.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-carte_desc_Edge-no_use_A-gat"
echo "Completed: Embed-carte_desc_Edge-no_use_A-gat"

python clustering2.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-carte_desc_Edge-normal_A-att"
echo "Completed: Embed-carte_desc_Edge-normal_A-att"

python clustering2.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-carte_desc_Edge-normal_A-gat"
echo "Completed: Embed-carte_desc_Edge-normal_A-gat"

python clustering2.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-carte_Edge-mlp_A-att"
echo "Completed: Embed-carte_Edge-mlp_A-att"

python clustering2.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-carte_Edge-mlp_A-gat"
echo "Completed: Embed-carte_Edge-mlp_A-gat"

python clustering2.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-carte_Edge-no_use_A-att"
echo "Completed: Embed-carte_Edge-no_use_A-att"

python clustering2.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-carte_Edge-no_use_A-gat"
echo "Completed: Embed-carte_Edge-no_use_A-gat"

python clustering2.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-carte_Edge-normal_A-att"
echo "Completed: Embed-carte_Edge-normal_A-att"

python clustering2.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-carte_Edge-normal_A-gat"
echo "Completed: Embed-carte_Edge-normal_A-gat"

python clustering2.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-ours_Edge-mlp_A-att"
echo "Completed: Embed-ours_Edge-mlp_A-att"

python clustering2.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-ours_Edge-mlp_A-gat"
echo "Completed: Embed-ours_Edge-mlp_A-gat"

python clustering2.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-ours_Edge-no_use_A-att"
echo "Completed: Embed-ours_Edge-no_use_A-att"

python clustering2.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-ours_Edge-no_use_A-gat"
echo "Completed: Embed-ours_Edge-no_use_A-gat"

python clustering2.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-ours_Edge-normal_A-att"
echo "Completed: Embed-ours_Edge-normal_A-att"

python clustering2.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-ours_Edge-normal_A-gat"
echo "Completed: Embed-ours_Edge-normal_A-gat"

python clustering2.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-ours2_Edge-mlp_A-att"
echo "Completed: Embed-ours2_Edge-mlp_A-att"

python clustering2.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-ours2_Edge-mlp_A-gat"
echo "Completed: Embed-ours2_Edge-mlp_A-gat"

python clustering2.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-ours2_Edge-no_use_A-att"
echo "Completed: Embed-ours2_Edge-no_use_A-att"

python clustering2.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-ours2_Edge-no_use_A-gat"
echo "Completed: Embed-ours2_Edge-no_use_A-gat"

python clustering2.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-ours2_Edge-normal_A-att"
echo "Completed: Embed-ours2_Edge-normal_A-att"

python clustering2.py --attention_map_dir "/storage/personal/eungyeop/experiments/attention_map/gpt2_mean/heart/Full/Embed-ours2_Edge-normal_A-gat"
echo "Completed: Embed-ours2_Edge-normal_A-gat"

echo "All analyses completed!"