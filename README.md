# N-Body CPU/GPU Performance Study

Starter repository for comparing a direct gravitational N-body simulation across
serial C++, OpenMP, naïve CUDA, and shared-memory tiled CUDA implementations.

The full scientific and benchmarking plan is in
[`N-Body_CPU_GPU_Project_Plan.md`](N-Body_CPU_GPU_Project_Plan.md).

## Current status

This initial milestone provides:

- a CMake-based C++ project;
- a shared command-line configuration and simulation data model;
- deterministic particle initialization;
- backend extension points for serial, OpenMP, and CUDA implementations;
- starter test, benchmark, Slurm, results, and visualization directories.

The force calculation and time integration are intentionally left for the next
milestone.

## Build locally

Requirements: CMake 3.20+ and a C++17 compiler. OpenMP is used automatically
when available.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build --output-on-failure
```

Run the starter executable:

```bash
./build/nbody --implementation serial --bodies 1024 --steps 100 --seed 12345
```

Use `./build/nbody --help` to see the initial command-line options. On a
multi-configuration Windows build, the executable may be under `build/Release/`.

## CUDA on CARC

CUDA source placeholders are included, but CUDA is disabled by default so the
project remains buildable on systems without the CUDA toolkit. After loading the
appropriate compiler and CUDA modules on CARC:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNBODY_ENABLE_CUDA=ON
cmake --build build -j
```

Update the module names and Slurm settings in `scripts/` after checking the
current CARC environment and course allocation.

## Repository layout

```text
include/nbody/       Shared public headers
src/                 C++ implementations and program entry point
src/cuda/            CUDA backend placeholders
tests/               Small correctness tests
scripts/             Slurm and benchmark helpers
results/             Generated benchmark data (ignored by Git)
visualization/       Plotting and animation work
```

## Suggested next milestone

Implement the serial acceleration calculation and velocity-Verlet step, then
add a two-body orbit test. Once that reference implementation is correct, the
parallel backends can replace only the acceleration routine.

