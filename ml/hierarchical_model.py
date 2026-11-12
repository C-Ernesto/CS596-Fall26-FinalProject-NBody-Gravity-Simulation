"""Barnes-Hut-style exact-near/learned-far force approximation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn


@dataclass
class OctreeNode:
    center: np.ndarray
    half_size: float
    indices: np.ndarray
    mass: float
    center_of_mass: np.ndarray
    children: dict[int, "OctreeNode"] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        return not self.children


def octant_for_position(position: np.ndarray, center: np.ndarray) -> int:
    return (
        int(position[0] >= center[0])
        | (int(position[1] >= center[1]) << 1)
        | (int(position[2] >= center[2]) << 2)
    )


def build_octree(
    positions: np.ndarray,
    masses: np.ndarray,
    leaf_capacity: int,
    maximum_depth: int,
) -> OctreeNode:
    minimum = positions.min(axis=0)
    maximum = positions.max(axis=0)
    center = 0.5 * (minimum + maximum)
    half_size = max(float((maximum - minimum).max()) * 0.5, 1.0e-12)
    half_size *= 1.000001

    def build(indices: np.ndarray, node_center: np.ndarray, node_half: float, depth: int) -> OctreeNode:
        node_masses = masses[indices]
        mass = float(node_masses.sum())
        center_of_mass = (
            positions[indices] * node_masses[:, None]
        ).sum(axis=0) / mass
        node = OctreeNode(
            center=node_center,
            half_size=node_half,
            indices=indices,
            mass=mass,
            center_of_mass=center_of_mass,
        )
        if len(indices) <= leaf_capacity or depth >= maximum_depth:
            return node

        octants: dict[int, list[int]] = {}
        for index in indices:
            octant = octant_for_position(positions[index], node_center)
            octants.setdefault(octant, []).append(int(index))
        if len(octants) == 1 and node_half <= 1.0e-14:
            return node

        child_half = 0.5 * node_half
        for octant, child_indices in octants.items():
            signs = np.array(
                [
                    1.0 if octant & 1 else -1.0,
                    1.0 if octant & 2 else -1.0,
                    1.0 if octant & 4 else -1.0,
                ]
            )
            child_center = node_center + signs * child_half
            node.children[octant] = build(
                np.asarray(child_indices, dtype=np.int64),
                child_center,
                child_half,
                depth + 1,
            )
        return node

    return build(np.arange(len(positions), dtype=np.int64), center, half_size, 0)


def build_barnes_hut_interactions(
    positions: torch.Tensor,
    masses: torch.Tensor,
    theta: float,
    leaf_capacity: int,
    maximum_depth: int,
) -> dict[str, torch.Tensor | int]:
    """Build target/source interactions with exact leaf particles and aggregate cells."""
    if theta < 0.0:
        raise ValueError("theta cannot be negative")
    positions_cpu = positions.detach().cpu().numpy().astype(np.float64, copy=False)
    masses_cpu = masses.detach().cpu().numpy().reshape(-1).astype(np.float64, copy=False)
    root = build_octree(positions_cpu, masses_cpu, leaf_capacity, maximum_depth)

    target_indices: list[int] = []
    source_positions: list[np.ndarray] = []
    source_masses: list[float] = []
    aggregate_flags: list[bool] = []

    for target_index, target_position in enumerate(positions_cpu):
        stack: list[tuple[OctreeNode, bool]] = [(root, True)]
        while stack:
            node, contains_target = stack.pop()
            if node.is_leaf:
                for source_index in node.indices:
                    if int(source_index) == target_index:
                        continue
                    target_indices.append(target_index)
                    source_positions.append(positions_cpu[source_index])
                    source_masses.append(float(masses_cpu[source_index]))
                    aggregate_flags.append(False)
                continue

            distance = float(np.linalg.norm(node.center_of_mass - target_position))
            node_size = 2.0 * node.half_size
            if (
                not contains_target
                and distance > 0.0
                and node_size / distance < theta
            ):
                target_indices.append(target_index)
                source_positions.append(node.center_of_mass)
                source_masses.append(node.mass)
                aggregate_flags.append(True)
                continue

            target_octant = octant_for_position(target_position, node.center)
            for octant, child in node.children.items():
                stack.append((child, contains_target and octant == target_octant))

    device = positions.device
    dtype = positions.dtype
    source_array = (
        np.asarray(source_positions, dtype=np.float64).reshape(-1, 3)
        if source_positions
        else np.empty((0, 3), dtype=np.float64)
    )
    return {
        "target_indices": torch.tensor(target_indices, dtype=torch.long, device=device),
        "source_positions": torch.as_tensor(
            source_array, dtype=dtype, device=device
        ),
        "source_masses": torch.tensor(source_masses, dtype=dtype, device=device),
        "is_aggregate": torch.tensor(aggregate_flags, dtype=torch.bool, device=device),
        "body_interactions": aggregate_flags.count(False),
        "aggregate_interactions": aggregate_flags.count(True),
    }


class HierarchicalForceModel(nn.Module):
    """Exact body interactions near leaves and learned aggregate far field."""

    def __init__(
        self,
        hidden_dim: int = 64,
        hidden_layers: int = 3,
        theta: float = 0.7,
        leaf_capacity: int = 16,
        maximum_depth: int = 24,
        edge_chunk_size: int = 262144,
    ) -> None:
        super().__init__()
        if theta < 0 or leaf_capacity <= 0 or maximum_depth <= 0 or edge_chunk_size <= 0:
            raise ValueError("invalid hierarchical model configuration")
        layers: list[nn.Module] = []
        for layer_index in range(hidden_layers):
            layers.append(nn.Linear(2 if layer_index == 0 else hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.radial_network = nn.Sequential(*layers)
        self.theta = theta
        self.leaf_capacity = leaf_capacity
        self.maximum_depth = maximum_depth
        self.edge_chunk_size = edge_chunk_size
        self.last_interaction_stats: dict[str, float | int] = {}

    def forward(
        self,
        positions: torch.Tensor,
        masses: torch.Tensor,
        epsilon: torch.Tensor,
        ptr: torch.Tensor,
    ) -> torch.Tensor:
        if masses.ndim == 1:
            masses = masses.unsqueeze(1)
        output = torch.zeros_like(positions)
        total_body = 0
        total_aggregate = 0
        build_seconds = 0.0
        interaction_seconds = 0.0

        for system_index in range(len(epsilon)):
            start = int(ptr[system_index].item())
            end = int(ptr[system_index + 1].item())
            system_positions = positions[start:end]
            system_masses = masses[start:end]
            build_started = time.perf_counter()
            interactions = build_barnes_hut_interactions(
                system_positions,
                system_masses,
                self.theta,
                self.leaf_capacity,
                self.maximum_depth,
            )
            build_seconds += time.perf_counter() - build_started
            total_body += int(interactions["body_interactions"])
            total_aggregate += int(interactions["aggregate_interactions"])

            targets = interactions["target_indices"]
            sources = interactions["source_positions"]
            source_masses = interactions["source_masses"]
            aggregates = interactions["is_aggregate"]
            system_output = torch.zeros_like(system_positions)
            interaction_started = time.perf_counter()
            for edge_start in range(0, len(targets), self.edge_chunk_size):
                edge_end = min(edge_start + self.edge_chunk_size, len(targets))
                target_chunk = targets[edge_start:edge_end]
                displacement = sources[edge_start:edge_end] - system_positions[target_chunk]
                softened_r2 = (
                    displacement.square().sum(dim=1)
                    + epsilon[system_index].square()
                )
                coefficient = source_masses[edge_start:edge_end] * softened_r2.pow(-1.5)
                aggregate_chunk = aggregates[edge_start:edge_end]
                if aggregate_chunk.any():
                    features = torch.stack(
                        [
                            torch.log(softened_r2[aggregate_chunk].clamp_min(1.0e-12)),
                            torch.log(epsilon[system_index].clamp_min(1.0e-12)).expand(
                                int(aggregate_chunk.sum().item())
                            ),
                        ],
                        dim=1,
                    )
                    learned_kernel = torch.exp(
                        self.radial_network(features).squeeze(1).clamp(-20.0, 20.0)
                    )
                    coefficient = coefficient.clone()
                    coefficient[aggregate_chunk] = (
                        source_masses[edge_start:edge_end][aggregate_chunk]
                        * learned_kernel
                    )
                messages = coefficient.unsqueeze(1) * displacement
                system_output.index_add_(0, target_chunk, messages)
            if positions.is_cuda:
                torch.cuda.synchronize(positions.device)
            interaction_seconds += time.perf_counter() - interaction_started
            output[start:end] = system_output

        self.last_interaction_stats = {
            "body_interactions": total_body,
            "aggregate_interactions": total_aggregate,
            "total_interactions": total_body + total_aggregate,
            "build_seconds": build_seconds,
            "interaction_seconds": interaction_seconds,
        }
        return output

    def load_pairwise_radial_network(self, pairwise_model: nn.Module) -> None:
        self.radial_network.load_state_dict(pairwise_model.radial_network.state_dict())
