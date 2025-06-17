#!/bin/bash

# Set OMP threads
export OMP_NUM_THREADS=10

# Run attentionmap.py for each checkpoint file
echo "Running attentionmap.py for all checkpoint files..."

python attentionmap.py --checkpoint_path "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:carte_desc_Edge:False_A:att.pt"
echo "Completed: carte_desc_Edge-False_A-att"

python attentionmap.py --checkpoint_path "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:carte_desc_Edge:False_A:gat.pt"
echo "Completed: carte_desc_Edge-False_A-gat"

python attentionmap.py --checkpoint_path "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:carte_desc_Edge:True_A:att.pt"
echo "Completed: carte_desc_Edge-True_A-att"

python attentionmap.py --checkpoint_path "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:carte_desc_Edge:True_A:gat.pt"
echo "Completed: carte_desc_Edge-True_A-gat"

python attentionmap.py --checkpoint_path "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:carte_Edge:False_A:att.pt"
echo "Completed: carte_Edge-False_A-att"

python attentionmap.py --checkpoint_path "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:carte_Edge:False_A:gat.pt"
echo "Completed: carte_Edge-False_A-gat"

python attentionmap.py --checkpoint_path "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:carte_Edge:True_A:att.pt"
echo "Completed: carte_Edge-True_A-att"

python attentionmap.py --checkpoint_path "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:carte_Edge:True_A:gat.pt"
echo "Completed: carte_Edge-True_A-gat"

python attentionmap.py --checkpoint_path "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:ours_Edge:False_A:att.pt"
echo "Completed: ours_Edge-False_A-att"

python attentionmap.py --checkpoint_path "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:ours_Edge:False_A:gat.pt"
echo "Completed: ours_Edge-False_A-gat"

python attentionmap.py --checkpoint_path "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:ours_Edge:True_A:att.pt"
echo "Completed: ours_Edge-True_A-att"

python attentionmap.py --checkpoint_path "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:ours_Edge:True_A:gat.pt"
echo "Completed: ours_Edge-True_A-gat"

python attentionmap.py --checkpoint_path "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:ours2_Edge:False_A:att.pt"
echo "Completed: ours2_Edge-False_A-att"

python attentionmap.py --checkpoint_path "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:ours2_Edge:False_A:gat.pt"
echo "Completed: ours2_Edge-False_A-gat"

python attentionmap.py --checkpoint_path "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:ours2_Edge:True_A:att.pt"
echo "Completed: ours2_Edge-True_A-att"

python attentionmap.py --checkpoint_path "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:ours2_Edge:True_A:gat.pt"
echo "Completed: ours2_Edge-True_A-gat"

echo "All analyses completed!"