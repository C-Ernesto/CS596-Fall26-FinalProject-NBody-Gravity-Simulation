#!/bin/bash
#SBATCH --job-name=nbody-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

# TODO: Add the CARC account, GPU constraint, and current compiler/CUDA modules.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=true
export OMP_PLACES=cores

srun ./build/nbody --implementation cuda_tiled --bodies 8192 --steps 100 \
    --block-size 256

