#!/bin/bash
PY=/home/eungyeop/anaconda3/envs/protollm/bin/python
cd /home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217
export WANDB_MODE=disabled

ROOT=/storage/personal/eungyeop/dataset/logs
QUEUE=$ROOT/case1_breast_T_queue.txt
LOCK=$ROOT/case1_breast_T_queue.lock
DRV=$ROOT/case1_breast_T_driver.log

# job = "T12:seed" / "T24:seed"
: > $QUEUE
for s in 42 44 46 48 50; do echo "T12:$s" >> $QUEUE; done
for s in 42 44 46 48 50; do echo "T24:$s" >> $QUEUE; done
: > $LOCK
: > $DRV

run_one () {
  local gpu=$1 grp=$2 seed=$3
  local SRC LOGDIR
  if [ "$grp" = "T12" ]; then
    SRC="breast_c1_dfs12 breast_c2_dfs12 breast_c3_dfs12 breast_c4_dfs12"
    LOGDIR=$ROOT/case1_breast_T12
  else
    SRC="breast_c1_dfs24 breast_c2_dfs24 breast_c3_dfs24 breast_c4_dfs24"
    LOGDIR=$ROOT/case1_breast_T24
  fi
  echo "[gpu$gpu] start $grp seed=$seed $(date)" >> $DRV
  CUDA_VISIBLE_DEVICES=$gpu $PY main_EEE.py \
    --exp_mode case1 --sampling_alpha 1.0 \
    --source_data $SRC \
    --embed_type carte_d256 --llm_model gemma-medical --input_dim 256 \
    --batch_size 128 --num_classes 2 \
    --soft_tau 0.5 \
    --base_dir breast_${grp}_case1_st05_20260907 \
    --random_seed $seed --run_tag s${seed}_${grp}_st05 \
    > $LOGDIR/seed${seed}.log 2>&1
  echo "[gpu$gpu] end $grp seed=$seed rc=$? $(date)" >> $DRV
}

worker () {
  local gpu=$1 job
  while true; do
    job=$(flock $LOCK -c "head -1 $QUEUE; sed -i '1d' $QUEUE")
    [ -z "$job" ] && break
    run_one $gpu "${job%%:*}" "${job##*:}"
  done
}

for g in 0 1 3 4; do worker $g & done
wait
echo "[driver] ALL DONE $(date)" >> $DRV
