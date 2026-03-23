#!/bin/bash
#SBATCH --job-name=tdgl_nofastscan
#SBATCH --output=%j_slurm.out
#SBATCH --error=%j_slurm.err
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=128G

# Packaged worker and CPU settings are controlled here.
module load julia

SCRIPT="current_sweep_thermal_nofastscan.jl"

if [ "${1:-}" = "--sequential" ]; then
    SCRIPT="current_sweep_thermal_sequential.jl"
    shift
    export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-8}"
    julia "$SCRIPT" "$@"
else
    julia -p 31 "$SCRIPT" "$@"
fi
