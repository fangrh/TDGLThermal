#!/bin/bash
#SBATCH --job-name=tdgl-review
#SBATCH --output=%j_slurm.out
#SBATCH --error=%j_slurm.err
#SBATCH --ntasks=35
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=64G

set -euo pipefail

module load julia
test -s config.yaml
test -s current_sweep_thermal_nofastscan.jl
julia -p 34 current_sweep_thermal_nofastscan.jl config.yaml
