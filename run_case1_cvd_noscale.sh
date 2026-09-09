#!/bin/bash
PY=/home/eungyeop/anaconda3/envs/protollm/bin/python
cd /home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217
export WANDB_MODE=disabled
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

ROOT=/storage/personal/eungyeop/dataset/logs
LOGDIR=$ROOT/case1_cvd_noscale
DRV=$ROOT/case1_cvd_noscale_driver.log
mkdir -p $LOGDIR
: > $DRV

SRC="Medicaldataset Cardiovascular_Disease_Dataset Erbil_Cardiovascular_Health_Dataset cardio_SAheart heart_failure_clinical_records"

run_one () {
  local gpu=$1 fd=$2 st=$3 tag=$4 seed=$5
  echo "[gpu$gpu] start $tag seed=$seed $(date)" >> $DRV
  CUDA_VISIBLE_DEVICES=$gpu $PY main_EEE.py \
    --exp_mode case1 --sampling_alpha 1.0 \
    --source_data $SRC \
    --embed_type carte_d256 --llm_model gemma-medical --input_dim 256 \
    --batch_size 128 --num_classes 2 \
    --feat_distance $fd --soft_tau $st \
    --base_dir cvd_case1_${tag}_noscale_20260908 \
    --random_seed $seed --run_tag s${seed}_cvd_${tag}_noscale \
    > $LOGDIR/${tag}_seed${seed}.log 2>&1
  echo "[gpu$gpu] end $tag seed=$seed rc=$? $(date)" >> $DRV
}

run_one 5 cosine 0.5 cos_st05 42 &
run_one 6 l2     0.5 l2_st05  42 &
wait
echo "[driver] ALL DONE $(date)" >> $DRV
