#!/usr/bin/env bash
# Restart after eicu shape-mismatch failure.
# Convert gpt2_mean pkl -> mmap in permissive mode (drop minority-shape samples
# in eicu so the rest converts), then register 3 sweeps and launch agents on
# GPU 0 / 4 / 5. Yamls already have --llm_model gpt2_mean --use_mmap_embeddings.
#
# Usage:
#   screen -dmS gpt2_v2 bash /home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217/run_gpt2_sweeps_v2.sh
set -e
cd /home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217

LOG=/tmp/run_gpt2_v2_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

PY=/home/eungyeop/anaconda3/envs/protollm/bin/python

echo "=== [$(date)] step 1: kill any stale wandb agents / screens ==="
for s in g0-mimic g4-eicu g5-hirid g0-mimic-gpt2 g4-eicu-gpt2 g5-hirid-gpt2; do
  screen -S "$s" -X quit 2>/dev/null || true
done
pkill -f "wandb agent" 2>/dev/null || true
sleep 3

echo "=== [$(date)] step 2: sanity check yamls have gpt2_mean + use_mmap ==="
for f in sweep/mortality/mortality_fm_{mimic,eicu,hirid}.yaml; do
  if ! grep -q "gpt2_mean" "$f" || ! grep -q "use_mmap_embeddings" "$f"; then
    echo "  [FATAL] $f missing gpt2_mean or use_mmap_embeddings"
    exit 1
  fi
  echo "  [ok] $f"
done

echo "=== [$(date)] step 3: convert gpt2_mean mmap in permissive mode ==="
"$PY" convert_embeddings_to_mmap.py --all --llm_model gpt2_mean --permissive

echo "=== [$(date)] step 4: sanity check eicu mmap has a reasonable sample count ==="
"$PY" - <<'PYEOF'
import json, os, sys
base = "/storage/personal/eungyeop/dataset/embedding/tabular_embeddings_carte_mmap/gpt2_mean"
ok = True
for name in ['mimic_mortality','eicu_mortality','hirid_mortality','support_mortality','zigong_mortality','sic_mortality']:
    meta = json.load(open(os.path.join(base, name, 'meta.json')))
    N, N_raw, dropped = meta['N'], meta.get('N_raw', meta['N']), meta.get('samples_dropped', 0)
    pct = 100.0 * N / max(N_raw, 1)
    tag = "OK" if pct > 50 else "LOW"
    print(f"  {name:20s} kept {N}/{N_raw} ({pct:.1f}%)  [{tag}]")
    if pct < 50:
        ok = False
sys.exit(0 if ok else 2)
PYEOF
rc=$?
if [ $rc -eq 2 ]; then
  echo "  [WARN] at least one dataset retained <50% of samples; proceeding anyway."
fi

echo "=== [$(date)] step 5: register sweeps ==="
mimic_id=$(wandb sweep sweep/mortality/mortality_fm_mimic.yaml 2>&1 | tee -a "$LOG" | grep "wandb agent" | awk '{print $NF}')
eicu_id=$( wandb sweep sweep/mortality/mortality_fm_eicu.yaml  2>&1 | tee -a "$LOG" | grep "wandb agent" | awk '{print $NF}')
hirid_id=$(wandb sweep sweep/mortality/mortality_fm_hirid.yaml 2>&1 | tee -a "$LOG" | grep "wandb agent" | awk '{print $NF}')
echo "GPU 0 mimic $mimic_id"
echo "GPU 4 eicu  $eicu_id"
echo "GPU 5 hirid $hirid_id"

if [ -z "$mimic_id" ] || [ -z "$eicu_id" ] || [ -z "$hirid_id" ]; then
  echo "  [FATAL] sweep registration failed. check the log above."
  exit 1
fi

echo "=== [$(date)] step 6: launch agents in detached screens (GPU 0 / 4 / 5) ==="
screen -dmS g0-mimic bash -lc "cd /home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217; CUDA_VISIBLE_DEVICES=0 wandb agent $mimic_id 2>&1 | tee /tmp/g0_mimic_gpt2.log"
screen -dmS g4-eicu  bash -lc "cd /home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217; CUDA_VISIBLE_DEVICES=4 wandb agent $eicu_id  2>&1 | tee /tmp/g4_eicu_gpt2.log"
screen -dmS g5-hirid bash -lc "cd /home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217; CUDA_VISIBLE_DEVICES=5 wandb agent $hirid_id 2>&1 | tee /tmp/g5_hirid_gpt2.log"

echo "=== [$(date)] done ==="
screen -ls || true
