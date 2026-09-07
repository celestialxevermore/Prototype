#!/bin/bash
# breast landmark ML baselines — T=12 / T=24, 코호트 C1~C4, baseline 5종
# full-shot only. 로그: logs/breast_lm/{dataset}_seed{S}.log
set -u
cd /home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217
PY=/home/eungyeop/anaconda3/envs/protollm/bin/python
LOGDIR=/home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217/logs/breast_lm
mkdir -p "$LOGDIR"

TS=(12 24)
CS=(c1 c2 c3 c4)
SEEDS=(42 43 44)

TOTAL=$(( ${#TS[@]} * ${#CS[@]} * ${#SEEDS[@]} ))
i=0
START=$(date +%s)

for T in "${TS[@]}"; do
for C in "${CS[@]}"; do
for S in "${SEEDS[@]}"; do
  i=$((i+1))
  DS="breast_${C}_dfs${T}"
  echo "[$i/$TOTAL] $(date '+%H:%M:%S')  $DS  seed=$S"
  $PY main_ml.py \
    --source_data "$DS" \
    --base_dir breast_lm \
    --random_seed "$S" \
    --baseline lr xgb mlp cat rf \
    --skip_few \
    --balance balanced \
    --hp_metric auprc \
    --test_size 0.2 \
    --des "landmark T=${T}" \
    > "$LOGDIR/${DS}_seed${S}.log" 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "    !! FAILED rc=$RC  -> $LOGDIR/${DS}_seed${S}.log"
    tail -5 "$LOGDIR/${DS}_seed${S}.log" | sed 's/^/       /'
  fi
done; done; done

echo
echo "===== ALL DONE  ($(( $(date +%s) - START ))s) ====="
echo "결과: /storage/personal/eungyeop/experiments/experiments/ml_baselines_breast_lm/"
find /storage/personal/eungyeop/experiments/experiments/ml_baselines_breast_lm -name '*.json' 2>/dev/null | wc -l | xargs echo "저장된 json 개수:"
