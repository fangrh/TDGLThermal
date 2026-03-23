#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$ROOT_DIR/sweep_nofastscan_$TIMESTAMP"
MODE="parallel"
SCRIPT_NAME="current_sweep_thermal_nofastscan.jl"
SBATCH_ARGS=()

if [ "${1:-}" = "--sequential" ]; then
    MODE="sequential"
    SCRIPT_NAME="current_sweep_thermal_sequential.jl"
    SBATCH_ARGS+=(--sequential)
    shift
fi

CONFIG_FILE="${1:-config.yaml}"

for path in "$ROOT_DIR/submit.sh" "$ROOT_DIR/$CONFIG_FILE" "$ROOT_DIR/$SCRIPT_NAME" "$ROOT_DIR/animation_data_utils.jl"; do
    if [ ! -f "$path" ]; then
        echo "Missing required file: $path" >&2
        exit 1
    fi
done

mkdir -p "$RUN_DIR"
cp "$ROOT_DIR/submit.sh" "$RUN_DIR/submit.sh"
cp "$ROOT_DIR/$CONFIG_FILE" "$RUN_DIR/config.yaml"
cp "$ROOT_DIR/$SCRIPT_NAME" "$RUN_DIR/$SCRIPT_NAME"
cp "$ROOT_DIR/animation_data_utils.jl" "$RUN_DIR/animation_data_utils.jl"

cd "$RUN_DIR"
echo "Created run directory: $RUN_DIR"
echo "Submitting with args: ${SBATCH_ARGS[*]}"
sbatch submit.sh "${SBATCH_ARGS[@]}"
