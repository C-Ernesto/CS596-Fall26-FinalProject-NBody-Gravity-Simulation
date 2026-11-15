# ML dataset and sparse force model

## Candidate full dataset

The fixed matrix in `ml/dataset_matrix.json` adds uniform cubes, Plummer-like
spheres, and colliding clusters to training. Validation uses unseen seeds, while
testing contains unseen particle counts and a completely held-out rotating-disk
family. Splits are assigned before generation, so trajectories cannot leak
between them.

Preview the matrix without generating data:

```powershell
.\.venv\Scripts\python.exe .\ml\generate_full_dataset.py --dry-run
```

Generate every raw CUDA shard, then validate and convert it to compressed
Parquet:

```powershell
.\.venv\Scripts\python.exe .\ml\generate_full_dataset.py
.\.venv\Scripts\python.exe .\ml\prepare_full_dataset.py
```

`prepare_full_dataset.py` checks every snapshot, writes a snapshot index, and
fits normalization statistics using only the training split. Use
`SnapshotDataset` from `ml/dataset.py` to retrieve complete variable-size
snapshots for a GNN.

Analyze the prepared dataset before training:

```powershell
.\.venv\Scripts\python.exe .\ml\analyze_dataset.py
```

This produces distribution summaries, snapshot-level conservation residuals,
close-encounter diagnostics, a Markdown report, and four plots under
`results/dataset_analysis/` and `visualization/output/dataset_analysis/`.

Install the CUDA-enabled PyTorch dependency and test the tensor pipeline:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements-ml.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_torch_dataset.py -v
```

Use `make_dataloader` from `ml/torch_dataset.py` for shard-aware,
particle-budgeted variable-size batches. The default force-model features are
normalized position and mass; velocity remains available separately for the
integrator and optional ablation studies.

Train and evaluate the first N=128 pairwise baseline:

```powershell
.\.venv\Scripts\python.exe .\ml\train_baseline.py --epochs 30
.\.venv\Scripts\python.exe .\ml\evaluate_baseline.py --split validation --body-counts 128
```

The baseline is translation invariant and rotation equivariant, but explicitly
evaluates every pair. It validates the ML pipeline and provides an accuracy
reference; it is not intended to provide a speedup over the CUDA solver.

Validate the normalized velocity-Verlet rollout and then run it with learned
forces:

```powershell
.\.venv\Scripts\python.exe .\ml\rollout_baseline.py --predictor exact --split validation --body-counts 128
.\.venv\Scripts\python.exe .\ml\rollout_baseline.py --predictor learned --split validation --body-counts 128
```

The exact predictor checks the Python integrator against saved FP64 CUDA states.
The learned predictor measures autoregressive position, velocity, force, energy,
and momentum errors at each saved checkpoint.

Evaluate the first scalable-force prototype:

```powershell
.\.venv\Scripts\python.exe .\ml\evaluate_hierarchical.py --hierarchical-checkpoint results\ml_hierarchical_geometry\best_model.pt --theta 0,0.3,0.5,0.7,1.0,1.2 --snapshots-per-case 2 --body-counts 128 --device cuda
.\.venv\Scripts\python.exe .\ml\evaluate_hierarchical.py --hierarchical-checkpoint results\ml_hierarchical_geometry\best_model.pt --split test --evaluation-group interpolation_counts --theta 0,0.5,0.7,1.0,1.2 --snapshots-per-case 1 --body-counts 768 --device cuda --output-dir results\hierarchical_geometry_768 --figure visualization\output\hierarchical_geometry_768.png
```

`HierarchicalForceModel` uses exact interactions for nearby leaf particles and
the pairwise baseline's learned radial function for accepted far-field
center-of-mass cells. Setting `theta=0` is the correctness control: it opens
every cell and matches direct force. Increasing `theta` reduces evaluated
interactions at the cost of force error. The readable Python tree builder is a
correctness reference; the compiled C++ backend removes most of its runtime
overhead. These sweeps still establish an accuracy-versus-work tradeoff rather
than a CUDA speedup.

Build the compiled OpenMP and CUDA interaction backends, then benchmark the complete
sparse pipeline, including hierarchy construction, transfers, neural
evaluation, and momentum projection:

```powershell
.\.venv\Scripts\python.exe .\setup_ml_extension.py build_ext --inplace
.\.venv\Scripts\python.exe .\ml\benchmark_hierarchical.py --backend cuda --device cuda --output-dir results\hierarchical_benchmark_cuda --figure visualization\output\hierarchical_benchmark_cuda.png
```

This runs one warm-up and five measured repetitions for N=128 through N=3072,
writes detailed and median CSV files under `results/hierarchical_benchmark_cuda/`,
and produces `visualization/output/hierarchical_benchmark_cuda.png`. The included
run retained 13.8% of all-pairs interactions at N=3072 with 1.38% relative L2
force error. Moving hierarchy construction and packing into C++ reduced N=3072
from 1943 ms to 127.1 ms; moving traversal and interaction-list creation onto
CUDA reduced it again to 16.3 ms. At N=2048 the CUDA-traversed method takes
12.1 ms, while the existing tiled-CUDA result amortizes to 2.14 ms per step.
Use `--backend python` for the reference, `cpp` for OpenMP traversal, `cuda` for
GPU traversal, or `auto` to select the fastest available compatible backend.

Run the paired large-N crossover experiment against one exact tiled-CUDA force
evaluation on the same initial state:

```powershell
.\.venv\Scripts\python.exe .\ml\benchmark_large_n.py --device cuda --body-counts 4096,8192,16384,32768,65536 --regenerate
```

On the development RTX 3060 Laptop GPU, sparse ML remained 1.14x slower at
N=32768 but became 1.49x faster at N=65536. At the crossover point it retained
1.25% of direct interactions and had 1.86% relative L2 force error. These are
paired, force-only measurements, but currently cover only one uniform-cube seed
per size and compare FP32 ML with exact FP64 CUDA. The next validation should
repeat the crossover across seeds and initial-condition families and test
trajectory stability.

The multi-family validation is now reproducible with:

```powershell
.\.venv\Scripts\python.exe .\ml\benchmark_crossover_matrix.py --device cuda --regenerate
```

This tests N=32768 and N=65536 using three seeds each for the uniform cube,
Plummer sphere, two-cluster collision, and rotating disk. At N=65536, ML was
faster for all uniform-cube and disk seeds but slower for every Plummer and
collision seed. Only the uniform cube combined speed with low error (1.86%);
the other families had 15%–21% relative force error. Concentrated systems also
caused severe irregular-tree traversal slowdown. The result is therefore a
distribution-dependent crossover. The next model work should train on diverse
large-N structured states, while the next CUDA work should target divergence in
deep irregular trees.

A controlled larger-N fine-tune can be resumed from the existing hierarchical
checkpoint without training on every available snapshot:

```powershell
.\.venv\Scripts\python.exe .\ml\train_hierarchical.py --initializer results\ml_hierarchical_geometry\best_model.pt --output-dir results\ml_hierarchical_large_n --figure visualization\output\ml_hierarchical_large_n_history.png --body-counts 512,1024,2048 --theta 0.7 --interaction-backend cuda --epochs 5 --learning-rate 5e-5 --minimum-learning-rate 1e-6 --max-snapshots 1 --max-particles 2048 --training-snapshots-per-case 8 --validation-snapshots-per-case 4 --device cuda
```

Sampling is deterministic within each initial-condition/count group. This run
improved pooled validation loss by 26.2%, but the repeated large-N matrix showed
that it specialized toward the collision family: collision force error fell to
about 4.5%, while uniform-cube error rose to 7%–8%, Plummer remained near 16%,
and the held-out disk rose to 22%–25%. It is an informative experimental
checkpoint, not the default model. The next training iteration should balance
families explicitly and select checkpoints using per-family validation metrics.

Re-evaluate it using the existing exact timings:

```powershell
.\.venv\Scripts\python.exe .\ml\benchmark_crossover_matrix.py --device cuda --checkpoint results\ml_hierarchical_large_n\best_model.pt --output-dir results\crossover_matrix_large_trained --figure visualization\output\crossover_matrix_large_trained.png --exact-results-dir results\crossover_matrix
```

The corrected training mode normalizes the Huber loss by each snapshot's RMS
target force, averages snapshots equally, and selects using per-family
validation results. A regression guard prevents an improved average from hiding
damage to one trained family:

```powershell
.\.venv\Scripts\python.exe .\ml\train_hierarchical.py --initializer results\ml_hierarchical_geometry\best_model.pt --output-dir results\ml_hierarchical_balanced_guarded --figure visualization\output\ml_hierarchical_balanced_guarded_history.png --body-counts 512,1024,2048 --theta 0.7 --interaction-backend cuda --epochs 10 --learning-rate 2e-5 --minimum-learning-rate 1e-6 --loss-mode snapshot_relative_huber --checkpoint-selection mean_family_relative_l2 --maximum-family-regression 0.15 --max-snapshots 1 --max-particles 2048 --training-snapshots-per-case 8 --validation-snapshots-per-case 4 --device cuda
```

This selected epoch 3. Compared with the original checkpoint, large-N Plummer
error improved about 30% and collision error improved 37%–40%; uniform was
1.82% at N=32768 and 2.28% at N=65536. The held-out disk still regressed by
11%–17%, so the guarded model is a useful structured-family candidate rather
than the new default. The disk remains excluded from model selection.

Map the large-N accuracy/speed tradeoff across tree opening angles with:

```powershell
.\.venv\Scripts\python.exe .\ml\benchmark_theta_sweep.py --device cuda --body-counts 32768 --theta 0.5,0.7,1.0 --seeds-per-case 1 --warmups 0 --repetitions 1
.\.venv\Scripts\python.exe .\ml\benchmark_theta_sweep.py --device cuda --body-counts 32768,65536 --theta 1.0 --seeds-per-case 3 --warmups 1 --repetitions 5 --output-dir results\theta_sweep_full_theta1 --figure visualization\output\theta_sweep_full_theta1.png
```

The pilot rejects `theta=0.5` as too slow. At `theta=1.0` and N=65536, sparse
ML is faster than exact CUDA for all four families, but only uniform cube also
has low force error (1.69% for the original checkpoint). Guarded Plummer and
collision are about 11%–12%, and the held-out disk is about 26%. This establishes
a broad performance crossover, but not a broad accuracy-and-speed crossover.
Keep `theta=0.7` as the conservative default and treat `theta=1.0` as the
measured high-throughput setting.

Compare both checkpoints and both operating angles inside velocity-Verlet with:

```powershell
.\.venv\Scripts\python.exe .\ml\run_rollout_matrix.py --device cuda --body-counts 1024,2048 --max-trajectories-per-family 4 --overwrite
```

This runs 175 steps for two validation trajectories per trained family/count
and four held-out disk trajectories per count. At N=2048, guarded `theta=1.0`
was 18%–44% faster than guarded `theta=0.7`, depending on family, but increased
final position RMSE by 48%–112%. Guarded `theta=0.7` roughly halved Plummer
position error relative to the original checkpoint and improved held-out disk
position error by 18%–33%. All runs remained finite and the worst mean energy
drift was about 0.20%. Saved tiled-CUDA checkpoints are the accuracy reference;
reported simulation milliseconds measure the ML rollout only.

Fine-tune the aggregate-cell radial network while leaving exact-near forces
fixed:

```powershell
.\.venv\Scripts\python.exe .\ml\train_hierarchical.py --epochs 10 --theta 1.0 --body-counts 128 --device cuda
.\.venv\Scripts\python.exe .\ml\evaluate_hierarchical.py --hierarchical-checkpoint results\ml_hierarchical_geometry\best_model.pt --theta 0,0.3,0.5,0.7,1.0,1.2 --body-counts 128 --device cuda --output-dir results\hierarchical_geometry_validation --figure visualization\output\hierarchical_geometry_validation.png
```

The model uses softened distance, softening, cell-size/distance, cell occupancy,
and RMS-radius/distance. Training reduced full N=128 validation MSE from
`2.94e-4` to `2.14e-4`. The unseen-N and held-out-family evaluations measure
how well this single sparse model transfers beyond its training snapshots.

Run the hierarchical model inside velocity-Verlet:

```powershell
.\.venv\Scripts\python.exe .\ml\rollout_baseline.py --predictor hierarchical --checkpoint results\ml_hierarchical_geometry\best_model.pt --theta 0.7 --split validation --body-counts 128 --device cuda --output-dir results\rollout_hierarchical --figure-dir visualization\output\rollout_hierarchical
```

After 175 steps, overall validation position RMSE was `1.96e-5`, relative
energy drift was `3.09e-5`, and held-out-disk position RMSE was `1.55e-4`.
The model always subtracts mass-weighted mean acceleration, which enforces zero
total internal force without changing relative accelerations or interaction count.

This is intentionally named a **candidate** full dataset. Its labels are FP64
tiled CUDA results sampled against naïve FP64 CUDA, but the metadata remains
`pre_serial_validation` and `serial_validated: false`. When the independent CPU
solver is added, a representative subset must be compared against it before
the dataset is frozen for final experiments.

Before freezing the final dataset:

1. validate a representative CUDA label sample against the independent serial
   implementation;
2. review the matrix size and physical parameter ranges;
3. record the final code commit, compiler, CUDA, and GPU metadata.
