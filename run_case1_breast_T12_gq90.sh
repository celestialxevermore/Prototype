#!/bin/bash
PY=/home/eungyeop/anaconda3/envs/protollm/bin/python
cd /home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217
export WANDB_MODE=disabled
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

ROOT=/storage/personal/eungyeop/dataset/logs
LOGDIR=$ROOT/case1_breast_T12_gq90
QUEUE=$ROOT/case1_breast_T12_gq90_queue.txt
LOCK=$ROOT/case1_breast_T12_gq90_queue.lock
DRV=$ROOT/case1_breast_T12_gq90_driver.log
mkdir -p $LOGDIR

: > $QUEUE
for s in 42 44 46 48 50; do echo "$s" >> $QUEUE; done
: > $LOCK
: > $DRV

SRC="breast_c1_dfs12 breast_c2_dfs12 breast_c3_dfs12 breast_c4_dfs12"

run_one () {
  local gpu=$1 seed=$2
  echo "[gpu$gpu] start seed=$seed $(date)" >> $DRV
  CUDA_VISIBLE_DEVICES=$gpu $PY main_EEE.py \
    --exp_mode case1 --sampling_alpha 1.0 \
    --source_data $SRC \
    --embed_type carte_d256 --llm_model gemma-medical --input_dim 256 \
    --batch_size 128 --num_classes 2 \
    --soft_tau 0.5 \
    --base_dir breast_T12_case1_st05_gq90_20260907 \
    --random_seed $seed --run_tag s${seed}_T12_st05_gq90 \
    > $LOGDIR/seed${seed}.log 2>&1
  echo "[gpu$gpu] end seed=$seed rc=$? $(date)" >> $DRV
}

worker () {
  local gpu=$1 job
  while true; do
    job=$(flock $LOCK -c "head -1 $QUEUE; sed -i '1d' $QUEUE")
    [ -z "$job" ] && break
    run_one $gpu "$job"
  done
}

for g in 0 1 2 3 4; do worker $g & done
wait
echo "[driver] ALL DONE $(date)" >> $DRV
