#!/bin/bash

# Set OMP threads
export OMP_NUM_THREADS=10

# Run attentionmap.py for each checkpoint file
echo "Running attentionmap.py for all checkpoint files..."
#!/bin/bash

# Carte_desc embedding 파일들
python attentionmap.py --checkpoint_dir /storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:carte_desc_Edge:mlp_A:att_20250617_173832.pt
python attentionmap.py --checkpoint_dir /storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:carte_desc_Edge:mlp_A:gat_20250617_173510.pt
python attentionmap.py --checkpoint_dir /storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:carte_desc_Edge:no_use_A:att_20250617_174040.pt
python attentionmap.py --checkpoint_dir /storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:carte_desc_Edge:no_use_A:gat_20250617_173739.pt
python attentionmap.py --checkpoint_dir /storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:carte_desc_Edge:normal_A:att_20250617_173937.pt
python attentionmap.py --checkpoint_dir /storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:carte_desc_Edge:normal_A:gat_20250617_173701.pt

# Carte embedding 파일들  
python attentionmap.py --checkpoint_dir /storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:carte_Edge:mlp_A:att_20250617_173809.pt
python attentionmap.py --checkpoint_dir /storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:carte_Edge:mlp_A:gat_20250617_173509.pt
python attentionmap.py --checkpoint_dir /storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:carte_Edge:no_use_A:att_20250617_174015.pt
python attentionmap.py --checkpoint_dir /storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:carte_Edge:no_use_A:gat_20250617_173716.pt
python attentionmap.py --checkpoint_dir /storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:carte_Edge:normal_A:att_20250617_173914.pt
python attentionmap.py --checkpoint_dir /storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:carte_Edge:normal_A:gat_20250617_173716.pt

# Ours embedding 파일들
python attentionmap.py --checkpoint_dir /storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:ours_Edge:mlp_A:att_20250617_173733.pt
python attentionmap.py --checkpoint_dir /storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:ours_Edge:mlp_A:gat_20250617_173512.pt
python attentionmap.py --checkpoint_dir /storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:ours_Edge:no_use_A:att_20250617_173833.pt
python attentionmap.py --checkpoint_dir /storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:ours_Edge:no_use_A:gat_20250617_173659.pt
python attentionmap.py --checkpoint_dir /storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:ours_Edge:normal_A:att_20250617_173803.pt
python attentionmap.py --checkpoint_dir /storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:ours_Edge:normal_A:gat_20250617_173623.pt

# Ours2 embedding 파일들
python attentionmap.py --checkpoint_dir /storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:ours2_Edge:mlp_A:att_20250617_173704.pt
python attentionmap.py --checkpoint_dir /storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:ours2_Edge:mlp_A:gat_20250617_173514.pt
python attentionmap.py --checkpoint_dir /storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:ours2_Edge:no_use_A:att_20250617_173840.pt
python attentionmap.py --checkpoint_dir /storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:ours2_Edge:no_use_A:gat_20250617_173628.pt
python attentionmap.py --checkpoint_dir /storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:ours2_Edge:normal_A:att_20250617_173753.pt
python attentionmap.py --checkpoint_dir /storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/heart/Full/Embed:ours2_Edge:normal_A:gat_20250617_173552.pt

echo "All attention map extractions completed!"