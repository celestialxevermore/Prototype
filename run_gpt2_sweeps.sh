#!/usr/bin/env bash
# Wait for the in-flight gemma mmap pipeline to finish, then convert gpt2_mean
# pkl -> mmap, patch sweep yamls to use gpt2_mean, register sweeps, and launch
# agents on GPU 0/4/5. Runs detached; user is asleep.
#
# Usage:
#   screen -dmS gpt2_pipeline bash /home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217/run_gpt2_sweeps.sh
#
# After wake-up: `screen -ls`, `tail /tmp/g{0,4,5}_*_gpt2.log`
set -e
cd /home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217

LOG=/tmp/run_gpt2_sweeps_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

PY=/home/eungyeop/anaconda3/envs/protollm/bin/python

echo "=== [$(date)] step 1: wait for gemma mmap_pipeline screen to finish ==="
while screen -ls 2>/dev/null | grep -q mmap_pipeline; do
  sleep 60
done
echo "[$(date)] gemma pipeline gone, proceeding"

echo "=== [$(date)] step 2: kill stale gemma-based sweep agents / screens ==="
for s in g0-mimic g4-eicu g5-hirid; do
  screen -S "$s" -X quit 2>/dev/null || true
done
pkill -f "wandb agent" 2>/dev/null || true
sleep 5

echo "=== [$(date)] step 3: add --llm_model gpt2_mean to every mortality yaml ==="
for f in sweep/mortality/mortality_fm_*.yaml; do
  if ! grep -q "gpt2_mean" "$f"; then
    sed -i '/- --use_mmap_embeddings/i\  - --llm_model\n  - gpt2_mean' "$f"
    echo "  patched $f"
  else
    echo "  skip $f (already has gpt2_mean)"
  fi
done

echo "=== [$(date)] step 4: convert gpt2_mean mmap (6 ICU, size-ascending) ==="
"$PY" convert_embeddings_to_mmap.py --all --llm_model gpt2_mean

echo "=== [$(date)] step 5: register sweeps ==="
mimic_id=$(wandb sweep sweep/mortality/mortality_fm_mimic.yaml 2>&1 | tee -a "$LOG" | grep "wandb agent" | awk '{print $NF}')
eicu_id=$( wandb sweep sweep/mortality/mortality_fm_eicu.yaml  2>&1 | tee -a "$LOG" | grep "wandb agent" | awk '{print $NF}')
hirid_id=$(wandb sweep sweep/mortality/mortality_fm_hirid.yaml 2>&1 | tee -a "$LOG" | grep "wandb agent" | awk '{print $NF}')
echo "GPU 0 mimic $mimic_id"
echo "GPU 4 eicu  $eicu_id"
echo "GPU 5 hirid $hirid_id"

echo "=== [$(date)] step 6: launch agents in detached screens ==="
screen -dmS g0-mimic bash -lc "cd /home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217; CUDA_VISIBLE_DEVICES=0 wandb agent $mimic_id 2>&1 | tee /tmp/g0_mimic_gpt2.log"
screen -dmS g4-eicu  bash -lc "cd /home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217; CUDA_VISIBLE_DEVICES=4 wandb agent $eicu_id  2>&1 | tee /tmp/g4_eicu_gpt2.log"
screen -dmS g5-hirid bash -lc "cd /home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217; CUDA_VISIBLE_DEVICES=5 wandb agent $hirid_id 2>&1 | tee /tmp/g5_hirid_gpt2.log"

echo "=== [$(date)] done ==="
screen -ls || true
