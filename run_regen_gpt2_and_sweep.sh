#!/usr/bin/env bash
# Use existing gpt2_mean pkls (under tabular_embeddings_carte/gpt2_mean/) as-is.
# Just mmap-convert them and launch sweeps on GPU 0/4/5.
# No regeneration.
#
# Usage:
#   screen -dmS gpt2_go bash /home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217/run_regen_gpt2_and_sweep.sh
set -e
cd /home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217

LOG=/tmp/gpt2_go_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

PY=/home/eungyeop/anaconda3/envs/protollm/bin/python
WANDB=/home/eungyeop/anaconda3/envs/protollm/bin/wandb

echo "=== [$(date)] step 0: kill stale screens/agents ==="
for s in g0-mimic g4-eicu g5-hirid g0-mimic-gpt2 g4-eicu-gpt2 g5-hirid-gpt2 gpt2_pipeline gpt2_v2 mmap_pipeline; do
  screen -S "$s" -X quit 2>/dev/null || true
done
pkill -f "wandb agent" 2>/dev/null || true
sleep 3

echo "=== [$(date)] step 1: convert existing gpt2_mean pkl -> mmap (--force) ==="
"$PY" convert_embeddings_to_mmap.py --all --llm_model gpt2_mean --force

echo "=== [$(date)] step 2: verify meta.json ==="
"$PY" - <<'PYEOF'
import json, os, sys
base = "/storage/personal/eungyeop/dataset/embedding/tabular_embeddings_carte_mmap/gpt2_mean"
ok = True
for name in ['mimic_mortality','eicu_mortality','hirid_mortality','support_mortality','zigong_mortality','sic_mortality']:
    path = os.path.join(base, name, 'meta.json')
    if not os.path.exists(path):
        print(f"  MISSING {path}"); ok = False; continue
    meta = json.load(open(path))
    N, N_raw = meta['N'], meta.get('N_raw', meta['N'])
    print(f"  {name:22s} N={N} (raw={N_raw})  shapes={meta['shapes']}")
sys.exit(0 if ok else 1)
PYEOF

echo "=== [$(date)] step 3: register sweeps ==="
mimic_id=$("$WANDB" sweep sweep/mortality/mortality_fm_mimic.yaml 2>&1 | tee -a "$LOG" | grep "wandb agent" | awk '{print $NF}')
eicu_id=$( "$WANDB" sweep sweep/mortality/mortality_fm_eicu.yaml  2>&1 | tee -a "$LOG" | grep "wandb agent" | awk '{print $NF}')
hirid_id=$("$WANDB" sweep sweep/mortality/mortality_fm_hirid.yaml 2>&1 | tee -a "$LOG" | grep "wandb agent" | awk '{print $NF}')
echo "GPU 0 mimic $mimic_id"
echo "GPU 4 eicu  $eicu_id"
echo "GPU 5 hirid $hirid_id"
if [ -z "$mimic_id" ] || [ -z "$eicu_id" ] || [ -z "$hirid_id" ]; then
  echo "[FATAL] sweep registration failed"
  exit 1
fi

echo "=== [$(date)] step 4: launch agents on GPU 0/4/5 ==="
screen -dmS g0-mimic bash -lc "cd /home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217; CUDA_VISIBLE_DEVICES=0 $WANDB agent $mimic_id 2>&1 | tee /tmp/g0_mimic_gpt2.log"
screen -dmS g4-eicu  bash -lc "cd /home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217; CUDA_VISIBLE_DEVICES=4 $WANDB agent $eicu_id  2>&1 | tee /tmp/g4_eicu_gpt2.log"
screen -dmS g5-hirid bash -lc "cd /home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217; CUDA_VISIBLE_DEVICES=5 $WANDB agent $hirid_id 2>&1 | tee /tmp/g5_hirid_gpt2.log"

echo "=== [$(date)] done ==="
screen -ls || true
