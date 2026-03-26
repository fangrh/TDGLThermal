#!/bin/bash
#SBATCH --job-name=package_archives
#SBATCH --output=%j_package_archives.out
#SBATCH --error=%j_package_archives.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --mem=16G

set -euo pipefail

THREADS="${SLURM_CPUS_PER_TASK:-16}"

if ! command -v pigz >/dev/null 2>&1; then
    echo "pigz is required but not found in PATH" >&2
    exit 1
fi

for folder in changeT10 continuation_scan_jobs_up10_2_002; do
    if [ ! -d "$folder" ]; then
        echo "Source folder not found: $folder" >&2
        exit 1
    fi
done

tar -I "pigz -p ${THREADS}" -cf changeT10.tar.gz changeT10
tar -I "pigz -p ${THREADS}" -cf continuation_scan_jobs_up10_2_002.tar.gz continuation_scan_jobs_up10_2_002

echo "Created:"
echo "  changeT10.tar.gz"
echo "  continuation_scan_jobs_up10_2_002.tar.gz"
