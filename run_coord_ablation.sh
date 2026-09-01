#!/bin/bash
# =====================================================================
# Coordinate ablation — wall-clock / memory / performance
#   1) cos    : pooled-CLS cosine        (alignment params = 0)
#   2) xattn  : cross-attention          (W_Q/W_K/W_V, source 학습 → target 고정)
#   3) fgw    : FGW (Ours)               (alignment params = 0)
#
#   target = heart, source = 6 cardio cohorts (main_SS.py default)
#   few_shot ∈ {0, 4, 8, 16, 32, 64}
#   coordinate 계산만 교체, 나머지(GAT/prototype graph/basis GNN/head/τ/K/T) 동일
# =====================================================================
set -u

REPO=/home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217
PY=/home/eungyeop/anaconda3/envs/protollm/bin/python
BASE_DIR=20260727
SEED=42

# --- 환경변수로 덮어쓸 수 있는 것들 -----------------------------------------
#   soft_tau 는 model_sig 에 안 들어가므로 값이 바뀌면 TAG 도 반드시 바꿀 것
SOFT_TAU=${SOFT_TAU:-1}
TAG=${TAG:-st${SOFT_TAU}}
RUN_TAG=coord_abl_${BASE_DIR}_${TAG}
PROF_ROOT=${PROF_ROOT:-/storage/personal/eungyeop/experiments/experiments/coord_ablation_${BASE_DIR}_${TAG}}
LOG_ROOT=${PROF_ROOT}/logs

# GPU 2 는 사용 금지
GPU_COS=${GPU_COS:-1}
GPU_XATTN=${GPU_XATTN:-3}
GPU_FGW=${GPU_FGW:-4}
declare -A GPU=( [cos]=${GPU_COS} [xattn]=${GPU_XATTN} [fgw]=${GPU_FGW} )
SHOTS=${SHOTS:-"0 4 8 16 32 64"}

# 3 프로세스 동시 → 64 core / 3
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export WANDB_MODE=disabled

# --- 세 방식 전부 동일하게 들어가는 모델/학습 하이퍼파라미터 ----------------
COMMON=(
  --base_dir "${BASE_DIR}"
  --run_tag  "${RUN_TAG}"
  --llm_model gpt2_mean --input_dim 768 --embed_type carte
  --n_graphs 12 --n_nodes 12 --graph_dim 768
  --num_basis_layers 2 --basis_type ind --attn_type gat_v1
  --struct_hidden_dim 192
  --fgw_alpha 1.5 --alpha 0.5 --vq_beta 0.3 --tau 0.5 --soft_tau "${SOFT_TAU}"
  --hs_reg 0.5 --entropy_reg 0.01
  --source_lr 0.002 --source_lr_few 0.0001 --dropout_rate 0.2
  --random_seed "${SEED}"
  --target_data heart
  --source_data Medicaldataset Cardiovascular_Disease_Dataset Heart_disease_statlog \
                Erbil_Cardiovascular_Health_Dataset cardio_SAheart heart_failure_clinical_records
  --support_resamples 3
)

mkdir -p "${LOG_ROOT}"

run_mode () {
  local MODE=$1
  local DEV=${GPU[$MODE]}
  # few_shot=0 을 먼저 → pretraining 수행 + multi-source report + zero-shot
  # 이후 shot 들은 같은 run_tag/des 의 best_joint.pt 를 재사용
  for SHOT in ${SHOTS}; do
    local OUT="${PROF_ROOT}/${MODE}/shot${SHOT}"
    mkdir -p "${OUT}"
    echo "[$(date +%H:%M:%S)] === ${MODE} | few_shot=${SHOT} | GPU ${DEV} ==="
    CUDA_VISIBLE_DEVICES=${DEV} \
    "${PY}" "${REPO}/coord_ablation_entry.py" \
      --coord_mode "${MODE}" \
      --profile_out "${OUT}" \
      --warmup_batches 10 --measure_batches 50 \
      -- \
      "${COMMON[@]}" \
      --des "coord_${MODE}_${TAG}" \
      --few_shot "${SHOT}" \
      > "${LOG_ROOT}/${MODE}_shot${SHOT}.log" 2>&1
    local RC=$?
    echo "[$(date +%H:%M:%S)] --- ${MODE} shot=${SHOT} done (exit=${RC}) ---"
  done
  echo "[$(date +%H:%M:%S)] ===== ${MODE} ALL DONE ====="
}

cd "${REPO}" || exit 1
for M in cos xattn fgw; do
  run_mode "${M}" > "${LOG_ROOT}/driver_${M}.log" 2>&1 &
done
wait
echo "[$(date +%H:%M:%S)] ALL MODES FINISHED -> ${PROF_ROOT}"
