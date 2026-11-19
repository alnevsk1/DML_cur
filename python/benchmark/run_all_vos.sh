#!/usr/bin/env bash
# Полный прогон VOS-бенчмарка (этап 4): XMem, Cutie, DeAOT — инициализация маской первого кадра.
# Только для WSL (bash). Из PowerShell:  wsl -d Ubuntu -- bash python/benchmark/run_all_vos.sh
# Параметры через переменные окружения (по умолчанию — протокол авторов):
#   SHORT_SIDE=480|768|-1   короткая сторона входа (480 — как в статьях; 768 — бюджет пикселей SAM 2)
#   PRECISION=fp32|bf16     fp32 — по умолчанию в eval-скриптах Cutie/DeAOT и в замерах FPS XMem
#   MODELS="xmem cutie_base"  подмножество моделей
# Можно прерывать (Ctrl+C) и запускать снова: готовые треки пропускаются.
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/dml_cur}"
export TQDM_DISABLE=1

SHORT_SIDE="${SHORT_SIDE:-480}"
PRECISION="${PRECISION:-fp32}"
read -r -a MODELS <<< "${MODELS:-deaot_t xmem cutie_base deaot_s r50_deaot_l}"

cd "$(dirname "$0")/.."            # -> python/
LOG_DIR=../runs/vos/logs
mkdir -p "$LOG_DIR"

declare -A REPO=(
  [xmem]=XMem
  [cutie_base]=Cutie
  [deaot_t]=aot-benchmark
  [deaot_s]=aot-benchmark
  [r50_deaot_l]=aot-benchmark
)

for model in "${MODELS[@]}"; do
  log="$LOG_DIR/${model}_s${SHORT_SIDE}_${PRECISION}.log"
  echo "=== $(date '+%F %T') $model / short side $SHORT_SIDE / $PRECISION -> $log"
  PYTHONPATH="third_party/${REPO[$model]}" \
    uv run --no-sync python -m benchmark.run_vos --model "$model" \
      --short-side "$SHORT_SIDE" --precision "$PRECISION" 2>&1 | tee -a "$log"
done

echo "=== $(date '+%F %T') evaluate"
uv run --no-sync python -m benchmark.evaluate 2>&1 | tee "$LOG_DIR/evaluate.log"
