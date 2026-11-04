#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_root"
julia_bin="${JULIA_BIN:-julia}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-1}"
export GKSwstype="${GKSwstype:-100}"
export NOFASTSCAN_SKIP_WORKERS=1
export NOFASTSCAN_SKIP_MAIN=1

mode="${1:-help}"
if (($#)); then shift; fi
case "$mode" in
  cpu)
    exec "$julia_bin" --project="$repo_root" --startup-file=no bench/benchmark_cpu.jl "$@" ;;
  workers)
    exec "$julia_bin" -p "${TDGL_WORKERS:-8}" --project="$repo_root" --startup-file=no bench/benchmark_worker_scaling.jl "$@" ;;
  imex)
    exec "$julia_bin" --project="$repo_root/experiments/imex" --startup-file=no bench/benchmark_imex.jl "$@" ;;
  gpu-links)
    exec "$julia_bin" --project="$repo_root/experiments/gpu_probe" --startup-file=no bench/gpu_link_probe.jl "$@" ;;
  gpu-rhs)
    exec "$julia_bin" --project="$repo_root/experiments/gpu_probe" --startup-file=no bench/benchmark_gpu_rhs.jl "$@" ;;
  gpu-solve)
    exec "$julia_bin" --project="$repo_root/experiments/gpu_probe" --startup-file=no bench/benchmark_gpu_solve.jl "$@" ;;
  run)
    export NOFASTSCAN_SKIP_MAIN=0
    export NOFASTSCAN_SKIP_WORKERS=0
    exec "$julia_bin" --project="$repo_root" --startup-file=no current_sweep_thermal_nofastscan.jl "$@" ;;
  *)
    printf '%s\n' 'Usage: bash bench/run_linux.sh {cpu|workers|imex|gpu-links|gpu-rhs|gpu-solve|run} [arguments]' 'Set JULIA_BIN, JULIA_DEPOT_PATH and output paths for your installation. Worker benchmark uses TDGL_WORKERS (default 8).'
    [[ "$mode" == help ]] || exit 2 ;;
esac
