#!/bin/bash
#SBATCH --job-name=tdgl_nofastscan
#SBATCH --output=%j_slurm.out
#SBATCH --error=%j_slurm.err
#SBATCH --ntasks=35
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=64G

# Packaged worker and CPU settings are controlled here.
module load julia

SCRIPT="current_sweep_thermal_nofastscan.jl"

if [ "${1:-}" = "--sequential" ]; then
    SCRIPT="current_sweep_thermal_sequential.jl"
    shift
    export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-8}"
    julia "$SCRIPT" "$@"
else
    SLURM_TASKS="${SLURM_NTASKS:-64}"
    JULIA_WORKERS=$((SLURM_TASKS - 1))
    if [ "$JULIA_WORKERS" -lt 0 ]; then
        JULIA_WORKERS=0
    fi
    julia -p "$JULIA_WORKERS" "$SCRIPT" "$@"
fi
