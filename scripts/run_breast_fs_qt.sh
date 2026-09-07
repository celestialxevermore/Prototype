#!/bin/bash
# breast landmark — quantile threshold 재실행
#   기존 run_breast_fs.sh 와 동일한 설정에서 --threshold_mode quantile 만 추가.
#   결과는 ml_baselines_breast_lm_fs/quantile/ 아래에 따로 저장한다 (기존 결과 보존).
# 사용법: run_breast_fs_qt.sh <GPU_ID> <QUEUE_FILE>
set -u
GPU=$1; Q=$2
cd /home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217
PY=/home/eungyeop/anaconda3/envs/protollm/bin/python
RES=/storage/personal/eungyeop/experiments/experiments/ml_baselines_breast_lm_fs/quantile
LOGDIR=logs/breast_fs_qt; mkdir -p "$LOGDIR"
i=0; ok=0; skip=0; START=$(date +%s)

while :; do
  JOB=$(flock "$Q.lock" -c "head -1 '$Q'; sed -i '1d' '$Q'")
  [ -z "$JOB" ] && break
  set -- $JOB; DS=$1; SEED=$2; MODE=$3; K=$4
  i=$((i+1))
  if [ "$MODE" = "full" ]; then FLAG="--skip_few"; TAG="full"; else FLAG="--skip_full"; TAG="k${K}"; fi

  if ls "$RES/$DS/args_seed:$SEED"/*/f${K}_*.json >/dev/null 2>&1; then
    skip=$((skip+1)); continue
  fi

  echo "[gpu${GPU} ${i}] $(date '+%H:%M:%S')  $DS seed=$SEED $TAG"
  CUDA_VISIBLE_DEVICES=$GPU $PY main_ml.py \
    --source_data "$DS" --base_dir breast_lm_fs/quantile \
    --random_seed "$SEED" --few_shot "$K" $FLAG \
    --baseline lr xgb mlp rf cat \
    --balance balanced --hp_metric auprc --test_size 0.2 \
    --threshold_mode quantile \
    --des "landmark ${DS##*dfs}mo ${TAG} qt" \
    > "$LOGDIR/${DS}_s${SEED}_${TAG}.log" 2>&1 \
    && ok=$((ok+1)) || echo "   !! FAIL  $LOGDIR/${DS}_s${SEED}_${TAG}.log"
done
echo "== gpu${GPU} 종료: 실행 ${ok} / 건너뜀 ${skip}  ($(( $(date +%s) - START ))s) =="
