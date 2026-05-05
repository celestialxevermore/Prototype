#!/usr/bin/env bash
# One-shot: add yaml flag → convert all 6 sources → register 3 sweeps →
# launch 3 agents in detached screens (GPU 0 / 4 / 5).
#
# Usage:
#   pkill -f "wandb agent"     # kill the existing mimic sweep first
#   screen -S all bash /home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217/run_mmap_sweeps.sh
#   (Ctrl-A D to detach, then sleep)
#
# After wake-up: `screen -ls` to list, `tail /tmp/g{0,4,5}_*.log` to check progress.

set -e
cd /home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217

LOG=/tmp/run_mmap_sweeps_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

PY=/home/eungyeop/anaconda3/envs/protollm/bin/python

echo "=== [$(date)] step 1: add --use_mmap_embeddings flag to every mortality yaml ==="
for f in sweep/mortality/mortality_fm_*.yaml; do
  grep -q -- '--use_mmap_embeddings' "$f" || sed -i '$a\  - --use_mmap_embeddings' "$f"
done

echo "=== [$(date)] step 2: convert all 6 sources to mmap (~50min) ==="
"$PY" convert_embeddings_to_mmap.py --all

echo "=== [$(date)] step 3: register sweeps ==="
mimic_id=$(wandb sweep sweep/mortality/mortality_fm_mimic.yaml 2>&1 | tee -a "$LOG" | grep "wandb agent" | awk '{print $NF}')
eicu_id=$( wandb sweep sweep/mortality/mortality_fm_eicu.yaml  2>&1 | tee -a "$LOG" | grep "wandb agent" | awk '{print $NF}')
hirid_id=$(wandb sweep sweep/mortality/mortality_fm_hirid.yaml 2>&1 | tee -a "$LOG" | grep "wandb agent" | awk '{print $NF}')
echo "GPU 0 mimic $mimic_id"
echo "GPU 4 eicu  $eicu_id"
echo "GPU 5 hirid $hirid_id"

echo "=== [$(date)] step 4: launch agents in detached screens ==="
screen -dmS g0-mimic bash -lc "cd /home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217; CUDA_VISIBLE_DEVICES=0 wandb agent $mimic_id 2>&1 | tee /tmp/g0_mimic.log"
screen -dmS g4-eicu  bash -lc "cd /home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217; CUDA_VISIBLE_DEVICES=4 wandb agent $eicu_id  2>&1 | tee /tmp/g4_eicu.log"
screen -dmS g5-hirid bash -lc "cd /home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217; CUDA_VISIBLE_DEVICES=5 wandb agent $hirid_id 2>&1 | tee /tmp/g5_hirid.log"

echo "=== [$(date)] done ==="
screen -ls || true
