#!/bin/bash
LOGDIR=/storage/personal/eungyeop/dataset/logs/case1_breast8_st05
PY=/home/eungyeop/anaconda3/envs/protollm/bin/python
cd /home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217
export WANDB_MODE=disabled

SRC="breast_c1_dfs12 breast_c1_dfs24 breast_c2_dfs12 breast_c2_dfs24 breast_c3_dfs12 breast_c3_dfs24 breast_c4_dfs12 breast_c4_dfs24"

QUEUE=$LOGDIR/queue.txt
LOCK=$LOGDIR/queue.txt.lock
printf '42\n44\n46\n48\n50\n' > $QUEUE
: > $LOCK

run_one () {
  local gpu=$1 seed=$2
  echo "[gpu$gpu] start seed=$seed $(date)" >> $LOGDIR/driver.log
  CUDA_VISIBLE_DEVICES=$gpu $PY main_EEE.py \
    --exp_mode case1 --sampling_alpha 1.0 \
    --source_data $SRC \
    --embed_type carte_d256 --llm_model gemma-medical --input_dim 256 \
    --batch_size 128 --num_classes 2 \
    --soft_tau 0.5 \
    --base_dir breast8_case1_st05_20260907 \
    --random_seed $seed --run_tag s${seed}_st05 \
    > $LOGDIR/seed${seed}.log 2>&1
  echo "[gpu$gpu] end seed=$seed rc=$? $(date)" >> $LOGDIR/driver.log
}

worker () {
  local gpu=$1
  while true; do
    local seed
    seed=$(flock $LOCK -c "head -1 $QUEUE; sed -i '1d' $QUEUE")
    [ -z "$seed" ] && break
    run_one $gpu $seed
  done
}

for g in 0 1 3 4; do worker $g & done
wait
echo "[driver] ALL DONE $(date)" >> $LOGDIR/driver.log
