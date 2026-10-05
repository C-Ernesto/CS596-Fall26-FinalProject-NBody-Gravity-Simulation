#!/bin/bash
set -euo pipefail

# Starter matrix; expand after each implementation passes correctness tests.
for bodies in 256 512 1024 2048 4096 8192 16384; do
    ./build/nbody --implementation serial --bodies "${bodies}" --steps 100 \
        --seed 12345 --output results/benchmarks.csv
done

