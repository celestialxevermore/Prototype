#!/bin/bash
PY=/home/eungyeop/anaconda3/envs/protollm/bin/python
cd /home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217
export WANDB_MODE=disabled
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

ROOT=/storage/personal/eungyeop/dataset/logs
LOGDIR=$ROOT/final_grid_20260908
QUEUE=$ROOT/final_grid_queue.txt
LOCK=$ROOT/final_grid_queue.lock
DRV=$ROOT/final_grid_driver.log
mkdir -p $LOGDIR
: > $QUEUE; : > $LOCK; : > $DRV

# job = dataset:featdist:softtau:seed
for seed in 42 44 46 48 50; do
  for ds in breast cvd; do
    for fd in cosine l2; do
      for st in 0.5 0.1; do
        echo "${ds}:${fd}:${st}:${seed}" >> $QUEUE
      done
    done
  done
done

BREAST_SRC="breast_c1_dfs12 breast_c2_dfs12 breast_c3_dfs12 breast_c4_dfs12"
CVD_SRC="Medicaldataset Cardiovascular_Disease_Dataset Erbil_Cardiovascular_Health_Dataset cardio_SAheart heart_failure_clinical_records"

run_one () {
  local gpu=$1 ds=$2 fd=$3 st=$4 seed=$5
  local SRC TAG
  if [ "$ds" = "breast" ]; then SRC="$BREAST_SRC"; else SRC="$CVD_SRC"; fi
  if [ "$fd" = "cosine" ]; then TAG="cos"; else TAG="l2"; fi
  if [ "$st" = "0.5" ]; then TAG="${TAG}_st05"; else TAG="${TAG}_st01"; fi
  local NAME="${ds}_${TAG}_seed${seed}"
  echo "[gpu$gpu] START $NAME $(date +%H:%M:%S)" >> $DRV
  CUDA_VISIBLE_DEVICES=$gpu $PY main_EEE.py \
    --exp_mode case1 --sampling_alpha 1.0 \
    --source_data $SRC \
    --embed_type carte_d256 --llm_model gemma-medical --input_dim 256 \
    --batch_size 128 --num_classes 2 \
    --feat_distance $fd --soft_tau $st \
    --base_dir ${ds}_case1_${TAG}_final_20260908 \
    --random_seed $seed --run_tag s${seed}_${ds}_${TAG}_final \
    > $LOGDIR/${NAME}.log 2>&1
  echo "[gpu$gpu] END   $NAME rc=$? $(date +%H:%M:%S)" >> $DRV
}

worker () {
  local gpu=$1 job
  while true; do
    job=$(flock $LOCK -c "head -1 $QUEUE; sed -i '1d' $QUEUE")
    [ -z "$job" ] && break
    IFS=':' read -r ds fd st seed <<< "$job"
    run_one $gpu "$ds" "$fd" "$st" "$seed"
  done
}

for g in 0 1 2 3 4 5 6; do worker $g & done
wait
echo "[driver] ALL DONE $(date)" >> $DRV
