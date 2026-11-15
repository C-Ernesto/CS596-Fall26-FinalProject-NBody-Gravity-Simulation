from __future__ import annotations

import unittest

import pandas as pd

from ml.rollout_baseline import select_trajectories


class RolloutSelectionTests(unittest.TestCase):
    def test_limit_applies_per_family_and_particle_count(self) -> None:
        rows = []
        for family in ("uniform_cube", "plummer_sphere"):
            for count in (1024, 2048):
                for seed in (2, 1):
                    rows.append(
                        {
                            "trajectory_uid": f"{family}-{count}-{seed}",
                            "initial_condition": family,
                            "particle_count": count,
                            "seed": seed,
                        }
                    )
        selected = select_trajectories(pd.DataFrame(rows), 1)
        self.assertEqual(len(selected), 4)
        self.assertTrue(all(uid.endswith("-1") for uid in selected))


if __name__ == "__main__":
    unittest.main()
