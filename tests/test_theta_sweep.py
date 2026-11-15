from __future__ import annotations

import argparse
import unittest

from ml.benchmark_theta_sweep import parse_models, parse_theta


class ThetaSweepTests(unittest.TestCase):
    def test_parse_theta(self) -> None:
        self.assertEqual(parse_theta("0.5,0.7,1.0"), [0.5, 0.7, 1.0])
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_theta("0.0,0.7")

    def test_parse_named_models(self) -> None:
        models = parse_models("original=first.pt,guarded=second.pt")
        self.assertEqual(list(models), ["original", "guarded"])
        self.assertEqual(str(models["guarded"]), "second.pt")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_models("missing-name-separator.pt")


if __name__ == "__main__":
    unittest.main()
