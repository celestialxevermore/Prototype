#!/bin/bash
PY=/home/eungyeop/anaconda3/envs/protollm/bin/python
cd /home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217
export WANDB_MODE=disabled
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

ROOT=/storage/personal/eungyeop/dataset/logs
LOGDIR=$ROOT/case1_breast_T12_noscale
QUEUE=$ROOT/case1_breast_T12_noscale_queue.txt
LOCK=$ROOT/case1_breast_T12_noscale_queue.lock
DRV=$ROOT/case1_breast_T12_noscale_driver.log
mkdir -p $LOGDIR

# job = "featdist:softtau:tag:seed"
: > $QUEUE
echo "cosine:0.5:cos_st05:42" >> $QUEUE
echo "cosine:0.1:cos_st01:42" >> $QUEUE
echo "l2:0.5:l2_st05:42"      >> $QUEUE
echo "l2:0.1:l2_st01:42"      >> $QUEUE
echo "cosine:0.5:cos_st05:44" >> $QUEUE
: > $LOCK
: > $DRV

SRC="breast_c1_dfs12 breast_c2_dfs12 breast_c3_dfs12 breast_c4_dfs12"

run_one () {
  local gpu=$1 fd=$2 st=$3 tag=$4 seed=$5
  echo "[gpu$gpu] start $tag seed=$seed $(date)" >> $DRV
  CUDA_VISIBLE_DEVICES=$gpu $PY main_EEE.py \
    --exp_mode case1 --sampling_alpha 1.0 \
    --source_data $SRC \
    --embed_type carte_d256 --llm_model gemma-medical --input_dim 256 \
    --batch_size 128 --num_classes 2 \
    --feat_distance $fd --soft_tau $st \
    --base_dir breast_T12_case1_${tag}_noscale_20260908 \
    --random_seed $seed --run_tag s${seed}_T12_${tag}_noscale \
    > $LOGDIR/${tag}_seed${seed}.log 2>&1
  echo "[gpu$gpu] end $tag seed=$seed rc=$? $(date)" >> $DRV
}

worker () {
  local gpu=$1 job
  while true; do
    job=$(flock $LOCK -c "head -1 $QUEUE; sed -i '1d' $QUEUE")
    [ -z "$job" ] && break
    IFS=':' read -r fd st tag seed <<< "$job"
    run_one $gpu "$fd" "$st" "$tag" "$seed"
  done
}

for g in 0 1 2 3 4; do worker $g & done
wait
echo "[driver] ALL DONE $(date)" >> $DRV
