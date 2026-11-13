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
interactions at the cost of force error. The current CPU/Python tree builder is
a research prototype and dominates runtime, so these sweeps establish an
accuracy-versus-work tradeoff rather than a CUDA speedup.

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
