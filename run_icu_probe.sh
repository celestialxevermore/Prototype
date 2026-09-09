#!/bin/bash
PY=/home/eungyeop/anaconda3/envs/protollm/bin/python
cd /home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217
export WANDB_MODE=disabled
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

LOGDIR=/storage/personal/eungyeop/dataset/logs/icu_probe_20260908
mkdir -p $LOGDIR
SRC="mimic_mortality eicu_mortality hirid_mortality support_mortality zigong_mortality sic_mortality"

run_one () {
  local gpu=$1 fd=$2 tag=$3
  CUDA_VISIBLE_DEVICES=$gpu $PY main_EEE.py \
    --exp_mode case1 --sampling_alpha 1.0 \
    --source_data $SRC \
    --embed_type carte_d256 --llm_model gemma-medical --input_dim 256 \
    --batch_size 128 --num_classes 2 \
    --feat_distance $fd --soft_tau 0.5 \
    --base_dir icu_case1_${tag}_probe_20260908 \
    --random_seed 42 --run_tag s42_icu_${tag}_probe \
    > $LOGDIR/${tag}_seed42.log 2>&1
  echo "[$tag] rc=$? $(date)" >> $LOGDIR/driver.log
}
run_one 4 cosine cos_st05 &
run_one 6 l2     l2_st05  &
wait
