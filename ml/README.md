# Pilot ML dataset

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

Before producing the final dataset:

1. validate a representative CUDA label sample against the independent serial
   implementation;
2. add additional initial-condition families;
3. include held-out particle counts and seeds;
4. define and store the final normalization scales;
5. replace CSV with a more compact format if dataset size becomes large.
