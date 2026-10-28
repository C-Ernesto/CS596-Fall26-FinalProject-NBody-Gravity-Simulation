# Machine-Learned Force Approximation for N-Body Simulation

This project compares exact direct gravitational N-body simulation on CPUs and GPUs, then tests whether a hierarchical machine-learning model can predict each particle's acceleration faster than direct all-pairs force calculation.

The exact O(N²) implementations will include serial C++, OpenMP, naïve CUDA, and shared-memory tiled CUDA. Their FP64 accelerations provide the ground truth for training and evaluating the learned model.

The full research and benchmarking plan is in [`N-Body_CPU_GPU_Project_Plan.md`](N-Body_CPU_GPU_Project_Plan.md).

## Research idea

A neural network applied to every particle pair would still be O(N²) and would probably be slower than the analytical gravity equation. The proposed model therefore uses a sparse hierarchical graph:

- particle nodes represent nearby interactions;
- spatial aggregate nodes summarize distant groups of particles;
- physics-informed messages use relative positions and masses; and
- the model outputs one acceleration vector for every particle.

The project will compare exact and learned methods on both CPU and GPU. It will measure end-to-end runtime, scaling with particle count, one-step acceleration error, trajectory error, energy drift, momentum drift, and generalization to unseen systems. A hybrid version may compute nearby interactions exactly while learning only the far-field contribution.

## Important limitation

The model is an approximation to a physical law that is already known. Its value must come from avoiding most pairwise interactions, not from replacing a cheap force equation with a more expensive neural network. A learned speedup counts only when the complete preprocessing, graph construction, transfer, and inference time is lower while the resulting simulation stays within a stated error tolerance.

Classical Barnes–Hut force approximation is a strongly recommended comparison if the schedule permits.

## Current status

The repository currently provides:

- a CMake-based C++17 project;
- shared body and simulation data structures;
- deterministic particle initialization;
- command-line parsing;
- extension points for serial, OpenMP, and CUDA force backends;
- CUDA source placeholders;
- one deterministic-initialization test; and
- starter Slurm, benchmark, result, and visualization directories.

The gravitational force calculation, velocity-Verlet integration, dataset pipeline, and learned model are not implemented yet.

## Build locally

Requirements: CMake 3.20+ and a C++17 compiler. OpenMP is used automatically when available.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build --output-on-failure
```

Run the starter executable:

```bash
./build/nbody --implementation serial --bodies 1024 --steps 100 --seed 12345
```

On a multi-configuration Windows build, the executable may be under `build/Release/`.

## CUDA build

CUDA is disabled by default so the CPU skeleton builds without a CUDA toolkit:

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DNBODY_ENABLE_CUDA=ON
cmake --build build-cuda --config Release
```

CARC module names and Slurm settings should be updated after checking the current cluster environment and course allocation.

## Planned ML workflow

1. Generate exact FP64 particle accelerations using the validated simulator.
2. Build train, validation, and test splits grouped by complete simulation seed or trajectory.
3. Construct sparse hierarchical particle graphs.
4. Train a PyTorch model to predict per-particle acceleration.
5. Integrate predicted acceleration using the same velocity-Verlet method.
6. Compare exact and learned force evaluation on CPU and GPU.
7. Test unseen particle counts, seeds, and initial-condition families.

The learned model will normally run in FP32. Exact FP64 results remain the reference.

## Repository layout

```text
include/nbody/       Shared C++ headers
src/                 C++ implementations and program entry point
src/cuda/            CUDA backend placeholders
tests/               Correctness tests
scripts/             Slurm and benchmark helpers
results/             Generated measurements (ignored by Git)
visualization/       Simulation and analysis figures
```

The planned ML additions are:

```text
ml/
├── dataset.py
├── hierarchy.py
├── model.py
├── train.py
├── evaluate.py
├── run_surrogate.py
└── models/
```

Generated datasets and model checkpoints will be excluded from Git except for small documented examples.

## Suggested next milestone

Implement the exact FP64 serial acceleration calculation and velocity-Verlet integration, then add analytical force and two-body orbit tests. After that reference is correct, add OpenMP and CUDA implementations and use the validated GPU solver to generate ML training labels.

