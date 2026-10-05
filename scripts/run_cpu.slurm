#!/bin/bash
#SBATCH --job-name=nbody-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

# TODO: Load the compiler module available on CARC.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=true
export OMP_PLACES=cores

srun ./build/nbody --implementation openmp --bodies 4096 --steps 100 \
    --threads "${SLURM_CPUS_PER_TASK}"

