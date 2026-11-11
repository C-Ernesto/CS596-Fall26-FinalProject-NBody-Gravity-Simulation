# Candidate dataset analysis

## Dataset size

- Body rows: 1,765,376
- Trajectories: 228
- Snapshots: 1,824
- Status: `pre_serial_validation`
- Serial validated: `false`

## Normalized acceleration by family

| Family | Median p50 | Maximum p99 | Maximum |
|---|---:|---:|---:|
| plummer_sphere | 0.279614 | 1.88465 | 29.8461 |
| rotating_disk | 1.72743 | 19.4211 | 37.6048 |
| two_cluster_collision | 1.27025 | 39.7823 | 267.606 |
| uniform_cube | 1.55324 | 31.9299 | 104.812 |

## Snapshot residual maxima

- Center-of-mass residual: 2.331221e-15
- Relative momentum residual: 1.052024e-15
- Relative net-force residual: 2.575193e-16

## Representative close encounters

- Smallest sampled distance/epsilon: 0.061065
- Largest sampled normalized acceleration: 267.606

The close-encounter calculation uses one deterministic representative snapshot and
the maximum-acceleration snapshot per split/evaluation/family/particle-count
combination. It is diagnostic, not an exhaustive all-snapshot nearest-neighbor search.
