#!/bin/bash
# =====================================================================
# Coordinate ablation (CORRECTED) — main_SS.py default HP + soft_tau=1 만 변경
#   1) cos    : pooled-CLS cosine        (alignment params = 0)
#   2) xattn  : cross-attention          (W_Q/W_K/W_V, source 학습 → target 고정)
#   3) fgw    : FGW (Ours) = baseline LCG.py
#
#   레퍼런스: source_to_source_test20260526 seed42 soft_tau=1
#            few_shot 8/32/64 = 0.8519 / 0.8641 / 0.8837
#   그 런의 HP 는 전부 argparse default (fgw_alpha 0.3 / alpha 0.7 / tau 0.5 /
#   vq_beta 0.3 / source_lr 1e-4 / dropout 0.1 / n_graphs 8 / n_nodes 8 /
#   hs_reg 0.1 / support_resamples 1) 이므로 여기서는 아무것도 지정하지 않는다.
#   명시하는 것은 데이터/임베딩/시드/soft_tau 뿐.
# =====================================================================
set -u

REPO=/home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217
PY=/home/eungyeop/anaconda3/envs/protollm/bin/python
BASE_DIR=20260727
SEED=42
TAG=g
RUN_TAG=coord_abl_${BASE_DIR}_${TAG}
PROF_ROOT=/storage/personal/eungyeop/experiments/experiments/coord_ablation_${BASE_DIR}_${TAG}
LOG_ROOT=${PROF_ROOT}/logs

# GPU 2 사용 금지. 현재 1/3/4 는 이전 런이 점유 중이라 0/5/6 을 쓴다.
declare -A GPU=( [cos]=0 [xattn]=5 [fgw]=6 )
SHOTS=${SHOTS:-"0 4 8 16 32 64"}

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export WANDB_MODE=disabled

# --- 세 방식 공통. main_SS.py default 를 덮어쓰는 것은 soft_tau 뿐 -----------
COMMON=(
  --base_dir "${BASE_DIR}"
  --run_tag  "${RUN_TAG}"
  --soft_tau 1
)

mkdir -p "${LOG_ROOT}"

wait_for_gpu () {   # 타이밍 측정이 목적이므로 GPU 를 독점한 뒤 시작
  local DEV=$1
  until [ "$(nvidia-smi -i ${DEV} --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 500 ]; do
    sleep 60
  done
}

run_mode () {
  local MODE=$1
  local DEV=${GPU[$MODE]}
  echo "[$(date +%H:%M:%S)] ${MODE}: GPU ${DEV} 비기를 대기"
  wait_for_gpu "${DEV}"
  echo "[$(date +%H:%M:%S)] ${MODE}: GPU ${DEV} 확보"
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
