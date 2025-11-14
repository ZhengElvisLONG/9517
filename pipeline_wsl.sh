#!/usr/bin/env bash
#
# pipeline_wsl.sh
# WSL-friendly pipeline for AgroPest-12 project.
#
# Usage:
#   ./pipeline_wsl.sh [all|yolov8_train|yolov8_eval|frcnn_train|frcnn_eval|visuals|check]
#
# Example:
#   ./pipeline_wsl.sh all
#

set -euo pipefail
IFS=$'\n\t'

########## Config (可按需修改) ##########
# Conda environment name (WSL 中常用)
CONDA_ENV="elvis"
# python executable if not using conda activation (optional)
PYTHON_BIN="python"

# Paths (相对于仓库根目录)
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${ROOT_DIR}/data"
CONFIGS_DIR="${ROOT_DIR}/configs"
SCRIPTS_DIR="${ROOT_DIR}/scripts"
EXPERIMENTS_DIR="${ROOT_DIR}/experiments"
REPORTS_DIR="${ROOT_DIR}/reports"

# YOLOv8 settings (示例)
YOLO_WEIGHTS="yolov8n.pt"                    # 相对或绝对路径
YOLO_IMG_SIZE=640
YOLO_BATCH=16
YOLO_EPOCHS=100
YOLO_CONF=0.25
YOLO_IOU=0.5
AGRO_YAML="${ROOT_DIR}/agropest.yaml"

# Faster R-CNN config file
FRCNN_CONFIG="${CONFIGS_DIR}/faster_rcnn_default.json"

# Logging
LOG_DIR="${EXPERIMENTS_DIR}/logs"
mkdir -p "${LOG_DIR}"
###########################################

timestamp() { date +"%Y%m%d_%H%M%S"; }

# Helper: print header
info() { echo -e "\n[INFO] $(date '+%F %T') - $*"; }
warn() { echo -e "\n[WARN] $(date '+%F %T') - $*"; }
err() { echo -e "\n[ERROR] $(date '+%F %T') - $*" >&2; }

# Detect WSL and convert Windows paths if needed (simple helper)
is_wsl=false
if grep -qi microsoft /proc/version 2>/dev/null; then is_wsl=true; fi

# Activate conda environment (works in WSL)
activate_conda_env() {
  if command -v conda >/dev/null 2>&1; then
    # Source conda functions (path may vary)
    # Try common locations
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
      source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
      source "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
      # fallback to generic
      eval "$(conda shell.bash hook)" || true
    fi

    # Activate
    info "Activating conda env: ${CONDA_ENV}"
    conda activate "${CONDA_ENV}"
    PYTHON_BIN="python"
  else
    warn "conda not found in PATH. Make sure the desired Python is available."
  fi
}

# GPU check
check_gpu() {
  info "Checking for GPU (nvidia-smi)..."
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  else
    warn "nvidia-smi not found. GPU may be unavailable. If you expect GPU, ensure NVIDIA drivers + WSL GPU support are installed."
  fi
}

# Ensure PYTHONPATH contains project root
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# Basic data sanity checks
check_data() {
  info "Running data sanity checks..."
  # Check directories
  for split in train valid test; do
    img_dir="${DATA_DIR}/${split}/images"
    lbl_dir="${DATA_DIR}/${split}/labels"
    if [ ! -d "${img_dir}" ]; then
      err "Missing images dir: ${img_dir}"
      return 1
    fi
    if [ ! -d "${lbl_dir}" ]; then
      warn "Missing labels dir: ${lbl_dir} (expected for YOLO)."
    fi
    # Count images
    img_count=$(find "${img_dir}" -maxdepth 1 -type f -iname "*.jpg" -o -iname "*.png" | wc -l || true)
    echo "  ${split}: ${img_count} images"
  done

  # check agropest.yaml
  if [ ! -f "${AGRO_YAML}" ]; then
    warn "agropest.yaml not found at ${AGRO_YAML}"
  else
    echo "Found dataset descriptor: ${AGRO_YAML}"
  fi
  info "Data sanity checks done."
}

# Run YOLOv8 train
run_yolov8_train() {
  info "Starting YOLOv8 training..."
  mkdir -p "${EXPERIMENTS_DIR}/yolov8"
  logf="${LOG_DIR}/yolov8_train_$(timestamp).log"

  "${PYTHON_BIN}" -u "${SCRIPTS_DIR}/train_yolov8.py" \
    --data "${AGRO_YAML}" \
    --model "${YOLO_WEIGHTS}" \
    --epochs "${YOLO_EPOCHS}" \
    --imgsz "${YOLO_IMG_SIZE}" \
    --batch "${YOLO_BATCH}" \
    --project "${EXPERIMENTS_DIR}/yolov8" \
    --name agropest_finetune \
    2>&1 | tee "${logf}"

  info "YOLOv8 training finished. Log: ${logf}"
}

# Run YOLOv8 evaluate
run_yolov8_eval() {
  info "Running YOLOv8 evaluation..."
  mkdir -p "${EXPERIMENTS_DIR}/yolov8"
  output_json="${EXPERIMENTS_DIR}/yolov8/val_metrics_$(timestamp).json"
  logf="${LOG_DIR}/yolov8_eval_$(timestamp).log"

  "${PYTHON_BIN}" -u "${SCRIPTS_DIR}/evaluate_yolov8.py" \
    --weights "${YOLO_WEIGHTS}" \
    --data "${AGRO_YAML}" \
    --split valid \
    --conf "${YOLO_CONF}" \
    --imgsz "${YOLO_IMG_SIZE}" \
    --iou-threshold "${YOLO_IOU}" \
    --save-json "${output_json}" \
    2>&1 | tee "${logf}"

  info "YOLOv8 evaluation finished. Metrics: ${output_json}, Log: ${logf}"
}

# Run Faster R-CNN train
run_frcnn_train() {
  info "Starting Faster R-CNN training..."
  mkdir -p "${EXPERIMENTS_DIR}/faster_rcnn"
  logf="${LOG_DIR}/frcnn_train_$(timestamp).log"

  "${PYTHON_BIN}" -u "${SCRIPTS_DIR}/train_faster_rcnn.py" \
    --config "${FRCNN_CONFIG}" \
    2>&1 | tee "${logf}"

  info "Faster R-CNN training finished. Log: ${logf}"
}

# Run Faster R-CNN evaluate
run_frcnn_eval() {
  info "Running Faster R-CNN evaluation..."
  mkdir -p "${EXPERIMENTS_DIR}/faster_rcnn"
  # try to find latest checkpoint if not provided
  CHECKPOINT="${EXPERIMENTS_DIR}/faster_rcnn/faster_rcnn_best.pth"
  if [ ! -f "${CHECKPOINT}" ]; then
    warn "Checkpoint ${CHECKPOINT} not found. If you want to evaluate other checkpoint, edit the script or move file."
  fi
  output_json="${EXPERIMENTS_DIR}/faster_rcnn/val_metrics_$(timestamp).json"
  logf="${LOG_DIR}/frcnn_eval_$(timestamp).log"

  "${PYTHON_BIN}" -u "${SCRIPTS_DIR}/evaluate_faster_rcnn.py" \
    --checkpoint "${CHECKPOINT}" \
    --split valid \
    --save-json "${output_json}" \
    2>&1 | tee "${logf}"

  info "Faster R-CNN evaluation finished. Metrics: ${output_json}, Log: ${logf}"
}

# Generate visuals
run_visuals() {
  info "Generating visuals & reports..."
  mkdir -p "${REPORTS_DIR}" "${REPORTS_DIR}/figures" "${REPORTS_DIR}/tables" "${REPORTS_DIR}/analysis"
  logf="${LOG_DIR}/generate_visuals_$(timestamp).log"

  "${PYTHON_BIN}" -u "${SCRIPTS_DIR}/generate_visuals.py" \
    2>&1 | tee "${logf}"

  info "Visuals generated. Check ${REPORTS_DIR}. Log: ${logf}"
}

# Print usage
usage() {
  cat <<EOF

Usage: $0 [all|check|yolov8_train|yolov8_eval|frcnn_train|frcnn_eval|visuals]

Commands:
  check           - env/data/gpu checks and minimal validation
  yolov8_train    - train YOLOv8 (calls scripts/train_yolov8.py)
  yolov8_eval     - evaluate YOLOv8 (calls scripts/evaluate_yolov8.py)
  frcnn_train     - train Faster R-CNN (calls scripts/train_faster_rcnn.py)
  frcnn_eval      - evaluate Faster R-CNN (calls scripts/evaluate_faster_rcnn.py)
  visuals         - generate reports (calls scripts/generate_visuals.py)
  all             - run check -> yolov8_train -> yolov8_eval -> frcnn_train -> frcnn_eval -> visuals

EOF
}

########## Main ##########
main_cmd="${1:-all}"

case "${main_cmd}" in
  check)
    activate_conda_env || true
    check_gpu || true
    check_data
    ;;
  yolov8_train)
    activate_conda_env || true
    check_gpu || true
    check_data
    run_yolov8_train
    ;;
  yolov8_eval)
    activate_conda_env || true
    check_gpu || true
    check_data
    run_yolov8_eval
    ;;
  frcnn_train)
    activate_conda_env || true
    check_gpu || true
    check_data
    run_frcnn_train
    ;;
  frcnn_eval)
    activate_conda_env || true
    check_gpu || true
    check_data
    run_frcnn_eval
    ;;
  visuals)
    activate_conda_env || true
    run_visuals
    ;;
  all)
    activate_conda_env || true
    check_gpu || true
    check_data
    run_yolov8_train
    run_yolov8_eval
    run_frcnn_train
    run_frcnn_eval
    run_visuals
    ;;
  *)
    usage
    exit 1
    ;;
esac

info "Pipeline '${main_cmd}' finished."
