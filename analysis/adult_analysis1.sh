#!/bin/bash

# Set OMP threads
export OMP_NUM_THREADS=10
#/storage/personal/eungyeop/experiments/experiments/checkpoints/gpt2_mean/adult/Full/Embed:carte_desc_Edge:mlp_A:att_20250617_173832.pt
# Run analysis1.py for all attention map directories
echo "Running analysis1.py for all attention map directories..."

python analysis1.py --checkpoint_dir "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/adult/Full/Embed-carte_desc_Edge-mlp_A-att"
echo "Completed: carte_desc_Edge-mlp_A-att"

python analysis1.py --checkpoint_dir "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/adult/Full/Embed-carte_desc_Edge-mlp_A-gat"
echo "Completed: carte_desc_Edge-mlp_A-gat"

python analysis1.py --checkpoint_dir "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/adult/Full/Embed:carte_desc_Edge:no_use_A:att_20250617_174040.pt"
echo "Completed: carte_desc_Edge-no_use_A-att"

python analysis1.py --checkpoint_dir "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/adult/Full/Embed:carte_desc_Edge:no_use_A:gat_20250617_173739.pt"
echo "Completed: carte_desc_Edge-no_use_A-gat"

python analysis1.py --checkpoint_dir "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/adult/Full/Embed:carte_desc_Edge:normal_A:att_20250617_173937.pt"
echo "Completed: carte_desc_Edge-normal_A-att"

python analysis1.py --checkpoint_dir "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/adult/Full/Embed:carte_desc_Edge:normal_A:gat_20250617_173701.pt"
echo "Completed: carte_desc_Edge-normal_A-gat"

python analysis1.py --checkpoint_dir "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/adult/Full/Embed:carte_Edge:mlp_A:att_20250617_173809.pt"
echo "Completed: carte_Edge-mlp_A-att"

python analysis1.py --checkpoint_dir "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/adult/Full/Embed:carte_Edge:mlp_A:gat_20250617_173509.pt"
echo "Completed: carte_Edge-mlp_A-gat"

python analysis1.py --checkpoint_dir "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/adult/Full/Embed:carte_Edge:no_use_A:att_20250617_174019.pt"
echo "Completed: carte_Edge-no_use_A-att"

python analysis1.py --checkpoint_dir "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/adult/Full/Embed:carte_Edge:no_use_A:gat_20250617_173716.pt"
echo "Completed: carte_Edge-no_use_A-gat"

python analysis1.py --checkpoint_dir "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/adult/Full/Embed:carte_Edge:normal_A:att_20250617_173914.pt"
echo "Completed: carte_Edge-normal_A-att"

python analysis1.py --checkpoint_dir "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/adult/Full/Embed:carte_Edge:normal_A:gat_20250617_173623.pt"
echo "Completed: carte_Edge-normal_A-gat"

python analysis1.py --checkpoint_dir "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/adult/Full/Embed:ours_Edge:mlp_A:att_20250617_173733.pt"
echo "Completed: ours_Edge-mlp_A-att"

python analysis1.py --checkpoint_dir "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/adult/Full/Embed:ours_Edge:mlp_A:gat_20250617_173512.pt"
echo "Completed: ours_Edge-mlp_A-gat"

python analysis1.py --checkpoint_dir "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/adult/Full/Embed:ours_Edge:no_use_A:att_20250617_173833.pt"
echo "Completed: ours_Edge-no_use_A-att"

python analysis1.py --checkpoint_dir "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/adult/Full/Embed:ours_Edge:no_use_A:gat_20250617_173659.pt"
echo "Completed: ours_Edge-no_use_A-gat"

python analysis1.py --checkpoint_dir "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/adult/Full/Embed:ours_Edge:normal_A:att_20250617_173803.pt"
echo "Completed: ours_Edge-normal_A-att"

python analysis1.py --checkpoint_dir "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/adult/Full/Embed:ours_Edge:normal_A:gat_20250617_173623.pt"
echo "Completed: ours_Edge-normal_A-gat"

python analysis1.py --checkpoint_dir "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/adult/Full/Embed:ours2_Edge:mlp_A:att_20250617_173704.pt"
echo "Completed: ours2_Edge-mlp_A-att"

python analysis1.py --checkpoint_dir "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/adult/Full/Embed:ours2_Edge:mlp_A:gat_20250617_173514.pt"
echo "Completed: ours2_Edge-mlp_A-gat"

python analysis1.py --checkpoint_dir "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/adult/Full/Embed:ours2_Edge:no_use_A:att_20250617_173840.pt"
echo "Completed: ours2_Edge-no_use_A-att"

python analysis1.py --checkpoint_dir "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/adult/Full/Embed:ours2_Edge:no_use_A:gat_20250617_173628.pt"
echo "Completed: ours2_Edge-no_use_A-gat"

python analysis1.py --checkpoint_dir "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/adult/Full/Embed:ours2_Edge:normal_A:att_20250617_173753.pt"
echo "Completed: ours2_Edge-normal_A-att"

python analysis1.py --checkpoint_dir "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/adult/Full/Embed:ours2_Edge:normal_A:gat_20250617_173552.pt"
echo "Completed: ours2_Edge-normal_A-gat"

echo "All analyses completed!"