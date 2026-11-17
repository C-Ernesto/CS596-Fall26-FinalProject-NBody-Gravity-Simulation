#!/bin/bash
set -euo pipefail

output="results/cpu_force_sweep.csv"
rm -f "${output}"
counts="256,512,1024,2048,4096,8192"
families=(uniform_cube plummer_sphere two_cluster_collision rotating_disk)
threads=(1 2 4 8 16)
append=()

for family_index in "${!families[@]}"; do
    family="${families[family_index]}"
    base_seed=$((700000 + 100000 * family_index))
    ./build/nbody_benchmark_force --implementations serial \
        --body-counts "${counts}" --base-seed "${base_seed}" \
        --trajectories-per-count 3 --repetitions 5 \
        --initial-condition "${family}" --output "${output}" "${append[@]}"
    append=(--append)
    for thread_count in "${threads[@]}"; do
        ./build/nbody_benchmark_force --implementations openmp \
            --threads "${thread_count}" --body-counts "${counts}" \
            --base-seed "${base_seed}" --trajectories-per-count 3 \
            --repetitions 5 --initial-condition "${family}" \
            --output "${output}" --append
    done
done

