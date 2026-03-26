#!/bin/bash
#SBATCH --job-name=tdgl_continue_scan2
#SBATCH --output=%j_slurm.out
#SBATCH --error=%j_slurm.err
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=128G

module load julia

julia -p 31 current_sweep_thermal_continue.jl "$@"
