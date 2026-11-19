#!/usr/bin/env bash
# Контрольный протокол VisDrone-SOT val (этап 4): 5 PVS-моделей × (box, mask) + 5 VOS-моделей × mask.
# Только для WSL (bash). Из PowerShell:  wsl -d Ubuntu -- bash python/benchmark/run_all_sot.sh
# Нужны маски инициализации: tools/visdrone_init_masks.py (data/annotations/visdrone_sot_val_init_masks.json).
# Можно прерывать (Ctrl+C) и запускать снова: готовые последовательности пропускаются.
# Не запускать одновременно с другими прогонами (замеры скорости).
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/dml_cur}"
export TQDM_DISABLE=1

cd "$(dirname "$0")/.."            # -> python/
LOG_DIR=../runs/sot/logs
mkdir -p "$LOG_DIR"

declare -A REPO=(
  [sam2.1_tiny]=sam2
  [sam2.1_small]=sam2
  [efficienttam_ti]=EfficientTAM
  [efficienttam_s]=EfficientTAM
  [edgetam]=EdgeTAM
  [xmem]=XMem
  [cutie_base]=Cutie
  [deaot_t]=aot-benchmark
  [deaot_s]=aot-benchmark
  [r50_deaot_l]=aot-benchmark
)
RUNS=(
  "sam2.1_tiny box" "sam2.1_tiny mask"
  "efficienttam_ti box" "efficienttam_ti mask"
  "sam2.1_small box" "sam2.1_small mask"
  "efficienttam_s box" "efficienttam_s mask"
  "edgetam box" "edgetam mask"
  "deaot_t mask" "xmem mask" "cutie_base mask" "deaot_s mask" "r50_deaot_l mask"
)

for run in "${RUNS[@]}"; do
  read -r model init <<< "$run"
  log="$LOG_DIR/${model}__${init}.log"
  echo "=== $(date '+%F %T') $model / $init -> $log"
  PYTHONPATH="third_party/${REPO[$model]}" \
    uv run --no-sync python -m benchmark.run_sot --model "$model" --init "$init" 2>&1 | tee -a "$log"
done

echo "=== $(date '+%F %T') evaluate"
uv run --no-sync python -m benchmark.evaluate_sot 2>&1 | tee "$LOG_DIR/evaluate.log"
