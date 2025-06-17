#!/bin/bash

# Set OMP threads
export OMP_NUM_THREADS=10

# Run label2.py for each clustering directory
echo "Running label2.py for all clustering directories..."

python label2.py --clustering_dir "/storage/personal/eungyeop/experiments/visualization/gpt2_mean/heart/Full/Embed-carte_desc_Edge-False_A-att/clustering_8"
echo "Completed: carte_desc_Edge-False_A-att"

python label2.py --clustering_dir "/storage/personal/eungyeop/experiments/visualization/gpt2_mean/heart/Full/Embed-carte_desc_Edge-False_A-gat/clustering_8"
echo "Completed: carte_desc_Edge-False_A-gat"

python label2.py --clustering_dir "/storage/personal/eungyeop/experiments/visualization/gpt2_mean/heart/Full/Embed-carte_desc_Edge-True_A-att/clustering_8"
echo "Completed: carte_desc_Edge-True_A-att"

python label2.py --clustering_dir "/storage/personal/eungyeop/experiments/visualization/gpt2_mean/heart/Full/Embed-carte_desc_Edge-True_A-gat/clustering_8"
echo "Completed: carte_desc_Edge-True_A-gat"

python label2.py --clustering_dir "/storage/personal/eungyeop/experiments/visualization/gpt2_mean/heart/Full/Embed-carte_Edge-False_A-att/clustering_8"
echo "Completed: carte_Edge-False_A-att"

python label2.py --clustering_dir "/storage/personal/eungyeop/experiments/visualization/gpt2_mean/heart/Full/Embed-carte_Edge-False_A-gat/clustering_8"
echo "Completed: carte_Edge-False_A-gat"

python label2.py --clustering_dir "/storage/personal/eungyeop/experiments/visualization/gpt2_mean/heart/Full/Embed-carte_Edge-True_A-att/clustering_8"
echo "Completed: carte_Edge-True_A-att"

python label2.py --clustering_dir "/storage/personal/eungyeop/experiments/visualization/gpt2_mean/heart/Full/Embed-carte_Edge-True_A-gat/clustering_8"
echo "Completed: carte_Edge-True_A-gat"

python label2.py --clustering_dir "/storage/personal/eungyeop/experiments/visualization/gpt2_mean/heart/Full/Embed-ours_Edge-False_A-att/clustering_8"
echo "Completed: ours_Edge-False_A-att"

python label2.py --clustering_dir "/storage/personal/eungyeop/experiments/visualization/gpt2_mean/heart/Full/Embed-ours_Edge-False_A-gat/clustering_8"
echo "Completed: ours_Edge-False_A-gat"

python label2.py --clustering_dir "/storage/personal/eungyeop/experiments/visualization/gpt2_mean/heart/Full/Embed-ours_Edge-True_A-att/clustering_8"
echo "Completed: ours_Edge-True_A-att"

python label2.py --clustering_dir "/storage/personal/eungyeop/experiments/visualization/gpt2_mean/heart/Full/Embed-ours_Edge-True_A-gat/clustering_8"
echo "Completed: ours_Edge-True_A-gat"

python label2.py --clustering_dir "/storage/personal/eungyeop/experiments/visualization/gpt2_mean/heart/Full/Embed-ours2_Edge-False_A-att/clustering_8"
echo "Completed: ours2_Edge-False_A-att"

python label2.py --clustering_dir "/storage/personal/eungyeop/experiments/visualization/gpt2_mean/heart/Full/Embed-ours2_Edge-False_A-gat/clustering_8"
echo "Completed: ours2_Edge-False_A-gat"

python label2.py --clustering_dir "/storage/personal/eungyeop/experiments/visualization/gpt2_mean/heart/Full/Embed-ours2_Edge-True_A-att/clustering_8"
echo "Completed: ours2_Edge-True_A-att"

python label2.py --clustering_dir "/storage/personal/eungyeop/experiments/visualization/gpt2_mean/heart/Full/Embed-ours2_Edge-True_A-gat/clustering_8"
echo "Completed: ours2_Edge-True_A-gat"

echo "All analyses completed!"