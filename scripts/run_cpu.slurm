#!/bin/bash
# CARC CPU thread sweep. Submit from the repository root; see project_overview.md.
#SBATCH --job-name=nbody-cpu
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

gcc_module="${NBODY_GCC_MODULE:-gcc/13.3.0}"
cmake_module="${NBODY_CMAKE_MODULE:-cmake/3.29.4}"
hardware_label="${NBODY_HARDWARE_LABEL:-cpu}"
body_counts="${NBODY_BODY_COUNTS:-256,512,1024,2048,4096,8192}"
thread_count_text="${NBODY_THREAD_COUNTS:-1 2 4 8 16}"
trajectories="${NBODY_TRAJECTORIES_PER_COUNT:-3}"
repetitions="${NBODY_REPETITIONS:-5}"

read -r -a thread_counts <<< "${thread_count_text}"
for threads in "${thread_counts[@]}"; do
    if (( threads > SLURM_CPUS_PER_TASK )); then
        echo "Requested ${threads} threads but only ${SLURM_CPUS_PER_TASK} CPUs were allocated." >&2
        exit 2
    fi
done

module purge
module load "${gcc_module}" "${cmake_module}"

cd "${SLURM_SUBMIT_DIR}"

build_dir="build-carc-cpu-${SLURM_JOB_ID}"
result_dir="results/carc"
output="${result_dir}/${hardware_label}-${SLURM_JOB_ID}-cpu-sweep.csv"
metadata="${result_dir}/${hardware_label}-${SLURM_JOB_ID}-metadata.txt"
mkdir -p "${result_dir}"

{
    echo "experiment=cpu_thread_sweep"
    echo "hardware_label=${hardware_label}"
    echo "slurm_job_id=${SLURM_JOB_ID}"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "body_counts=${body_counts}"
    echo "thread_counts=${thread_count_text}"
    echo "trajectories_per_count=${trajectories}"
    echo "repetitions=${repetitions}"
    date --iso-8601=seconds
    hostname
    lscpu
    module list
    gcc --version
    cmake --version
} > "${metadata}" 2>&1

cmake -S . -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DNBODY_ENABLE_CUDA=OFF \
    -DNBODY_BUILD_TESTS=ON
cmake --build "${build_dir}" --parallel "${SLURM_CPUS_PER_TASK}"
ctest --test-dir "${build_dir}" --output-on-failure

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
    base_seed=$((700000 + 100000 * family_index))

    srun --cpu-bind=cores "${build_dir}/nbody_benchmark_force" \
        --implementations serial \
        --body-counts "${body_counts}" \
        --base-seed "${base_seed}" \
        --trajectories-per-count "${trajectories}" \
        --repetitions "${repetitions}" \
        --initial-condition "${family}" \
        --output "${output}" \
        "${append[@]}"
    append=(--append)

    for threads in "${thread_counts[@]}"; do
        export OMP_NUM_THREADS="${threads}"
        srun --cpu-bind=cores "${build_dir}/nbody_benchmark_force" \
            --implementations openmp \
            --threads "${threads}" \
            --body-counts "${body_counts}" \
            --base-seed "${base_seed}" \
            --trajectories-per-count "${trajectories}" \
            --repetitions "${repetitions}" \
            --initial-condition "${family}" \
            --output "${output}" \
            --append
    done
done

echo "CPU sweep written to: ${output}"
echo "Hardware metadata written to: ${metadata}"
