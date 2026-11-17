#!/bin/bash
#SBATCH --job-name=nbody-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

# TODO: Load the compiler and CMake modules available on CARC.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=true
export OMP_PLACES=cores

mkdir -p results
rm -f results/cpu_force_sweep_slurm.csv

append=()
for threads in 1 2 4 8 16; do
    srun ./build/nbody_benchmark_force \
        --implementations openmp \
        --body-counts 256,512,1024,2048,4096,8192 \
        --base-seed 700000 \
        --trajectories-per-count 3 \
        --repetitions 5 \
        --initial-condition uniform_cube \
        --threads "${threads}" \
        --output results/cpu_force_sweep_slurm.csv \
        "${append[@]}"
    append=(--append)
done

