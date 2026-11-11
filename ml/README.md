# ML datasets

The pilot dataset uses the validated FP64 tiled CUDA solver for acceleration
labels. Every sampled snapshot can also be checked against naïve CUDA. This is
an infrastructure and sanity-check dataset, not the final training corpus.

Build the CUDA project, then generate the default pilot:

```powershell
cmake --build build-cuda-explicit --config Release
.\build-cuda-explicit\Release\nbody_generate_dataset.exe
```

The default generator creates eight independent uniform-cube trajectories:

- particle counts: 128 and 256;
- four seeds per particle count;
- four snapshots per trajectory;
- ten integration steps between snapshots;
- naïve-versus-tiled verification for every snapshot.

Prepare leakage-safe splits with the project virtual environment:

```powershell
.\.venv\Scripts\python.exe .\ml\prepare_dataset.py
```

Generated files are ignored by Git:

```text
data/raw/pilot_cuda.csv
data/processed/pilot/train.csv
data/processed/pilot/validation.csv
data/processed/pilot/test.csv
data/processed/pilot/manifest.csv
```

The splitter assigns complete trajectories to exactly one split. It never
mixes rows or snapshots from the same trajectory across training, validation,
and testing.

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

Compare it with zero, training-mean, and interaction-free local-MLP baselines:

```powershell
.\.venv\Scripts\python.exe .\ml\train_baseline.py --model local_mlp --epochs 30 --output-dir results\ml_local_baseline --figure visualization\output\ml_local_baseline_history.png
.\.venv\Scripts\python.exe .\ml\evaluate_simple_baselines.py --split validation --body-counts 128 --output results\simple_baselines\validation_metrics.csv
.\.venv\Scripts\python.exe .\ml\compare_baselines.py
```

The local MLP only sees each body's own position and mass. Its failure relative
to the pairwise model is the control showing that accurate acceleration requires
information about other particles rather than only initial-condition bias.

Validate the normalized velocity-Verlet rollout and then run it with learned
forces:

```powershell
.\.venv\Scripts\python.exe .\ml\rollout_baseline.py --predictor exact --split validation --body-counts 128
.\.venv\Scripts\python.exe .\ml\rollout_baseline.py --predictor learned --split validation --body-counts 128
```

The exact predictor checks the Python integrator against saved FP64 CUDA states.
The learned predictor measures autoregressive position, velocity, force, energy,
and momentum errors at each saved checkpoint.

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
