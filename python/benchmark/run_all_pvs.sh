#!/usr/bin/env bash
# Полный прогон PVS-бенчмарка (этап 4): 5 моделей × 3 варианта инициализации, затем оценка.
# Только для WSL (bash). Из PowerShell:  wsl -d Ubuntu -- bash python/benchmark/run_all_pvs.sh
# Можно прерывать (Ctrl+C) и запускать снова: готовые треки пропускаются.
# Порядок: сначала mask для всех моделей, затем click, затем box — частичные результаты полезны раньше.
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/dml_cur}"
export TQDM_DISABLE=1

cd "$(dirname "$0")/.."            # -> python/
LOG_DIR=../runs/pvs/logs
mkdir -p "$LOG_DIR"

declare -A REPO=(
  [sam2.1_tiny]=sam2
  [sam2.1_small]=sam2
  [efficienttam_ti]=EfficientTAM
  [efficienttam_s]=EfficientTAM
  [edgetam]=EdgeTAM
)
MODELS=(sam2.1_tiny efficienttam_ti edgetam efficienttam_s sam2.1_small)

for init in mask click box; do
  for model in "${MODELS[@]}"; do
    log="$LOG_DIR/${model}__${init}.log"
    echo "=== $(date '+%F %T') $model / $init -> $log"
    PYTHONPATH="third_party/${REPO[$model]}" \
      uv run --no-sync python -m benchmark.run_pvs --model "$model" --init "$init" 2>&1 | tee -a "$log"
  done
done

echo "=== $(date '+%F %T') evaluate"
uv run --no-sync python -m benchmark.evaluate 2>&1 | tee "$LOG_DIR/evaluate.log"
