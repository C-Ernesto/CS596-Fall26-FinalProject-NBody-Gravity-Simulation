#!/bin/bash
# CARC paired serial/OpenMP/tiled-CUDA crossover benchmark.
# Submit from the repository root; request a specific GPU on the sbatch command.
#SBATCH --job-name=nbody-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

gcc_module="${NBODY_GCC_MODULE:-gcc/13.3.0}"
cmake_module="${NBODY_CMAKE_MODULE:-cmake/3.29.4}"
cuda_module="${NBODY_CUDA_MODULE:-cuda/12.6.3}"
hardware_label="${NBODY_HARDWARE_LABEL:-gpu}"
body_counts="${NBODY_BODY_COUNTS:-256,512,1024,2048,4096,8192,16384}"
threads="${NBODY_THREADS:-16}"
block_size="${NBODY_BLOCK_SIZE:-64}"
trajectories="${NBODY_TRAJECTORIES_PER_COUNT:-3}"
repetitions="${NBODY_REPETITIONS:-5}"
cuda_architectures="${NBODY_CUDA_ARCHITECTURES:-native}"

if (( threads > SLURM_CPUS_PER_TASK )); then
    echo "Requested ${threads} threads but only ${SLURM_CPUS_PER_TASK} CPUs were allocated." >&2
    exit 2
fi

module purge
module load "${gcc_module}" "${cmake_module}" "${cuda_module}"

cd "${SLURM_SUBMIT_DIR}"

build_dir="build-carc-gpu-${SLURM_JOB_ID}"
result_dir="results/carc"
output="${result_dir}/${hardware_label}-${SLURM_JOB_ID}-crossover.csv"
metadata="${result_dir}/${hardware_label}-${SLURM_JOB_ID}-metadata.txt"
mkdir -p "${result_dir}"

{
    echo "experiment=paired_cpu_cuda_crossover"
    echo "hardware_label=${hardware_label}"
    echo "slurm_job_id=${SLURM_JOB_ID}"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "body_counts=${body_counts}"
    echo "openmp_threads=${threads}"
    echo "cuda_block_size=${block_size}"
    echo "cuda_architectures=${cuda_architectures}"
    echo "trajectories_per_count=${trajectories}"
    echo "repetitions=${repetitions}"
    date --iso-8601=seconds
    hostname
    lscpu
    nvidia-smi
    module list
    gcc --version
    cmake --version
    nvcc --version
} > "${metadata}" 2>&1

cmake -S . -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DNBODY_ENABLE_CUDA=ON \
    -DNBODY_BUILD_TESTS=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_architectures}"
cmake --build "${build_dir}" --parallel "${SLURM_CPUS_PER_TASK}"
ctest --test-dir "${build_dir}" --output-on-failure

export OMP_NUM_THREADS="${threads}"
export OMP_DYNAMIC=false
export OMP_PROC_BIND=true
export OMP_PLACES=cores

families=(
    uniform_cube
    plummer_sphere
    two_cluster_collision
    rotating_disk
)
append=()

for family_index in "${!families[@]}"; do
    family="${families[family_index]}"
    base_seed=$((900000 + 100000 * family_index))

    srun --cpu-bind=cores "${build_dir}/nbody_benchmark_force" \
        --implementations serial,openmp,cuda_tiled \
        --body-counts "${body_counts}" \
        --base-seed "${base_seed}" \
        --trajectories-per-count "${trajectories}" \
        --repetitions "${repetitions}" \
        --threads "${threads}" \
        --block-size "${block_size}" \
        --initial-condition "${family}" \
        --output "${output}" \
        "${append[@]}"
    append=(--append)
done

echo "Paired crossover written to: ${output}"
echo "Hardware metadata written to: ${metadata}"
