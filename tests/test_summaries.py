import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.attnlib.metrics import paired_structure_difference
from src.experiments.e8_heldout import summarise, plot_depth, plot_factors


def result(gap, excess=None):
    excess = gap if excess is None else excess
    band = {"per_example": {
        "matched_minus_random": gap,
        "same_minus_shuffled": excess,
        "gap_same_minus_shuffled": excess,
    }}
    return {"n_examples": len(gap), "symmetric_bands": {key: band for key in ("early", "late", "all")}}


class SummaryTests(unittest.TestCase):
    def test_structure_uses_paired_differences_and_withholds_control_cis(self):
        ordinary = result([.1, .2, .8, .9])
        repetition = result([.2, .3, .9, 1.])
        summary = paired_structure_difference(ordinary, repetition, n_bootstrap=100)
        for band in summary["bands"].values():
            estimate = band["matched_minus_random"]
            self.assertAlmostEqual(estimate["mean_change"], .1)
            np.testing.assert_allclose(estimate["confidence_interval"], [.1, .1])
            self.assertIsNone(band["same_minus_shuffled"]["confidence_interval"])
            self.assertIsNone(band["gap_same_minus_shuffled"]["confidence_interval"])

    def test_structure_missing_values_and_misalignment(self):
        ordinary, repeated = result([.1, .2]), result([.3, .4])
        repeated["symmetric_bands"]["late"] = {"per_example": {}}
        output = paired_structure_difference(ordinary, repeated)
        self.assertIsNone(output["bands"]["late"]["matched_minus_random"]["mean_change"])
        with self.assertRaises(ValueError):
            paired_structure_difference(ordinary, result([.1]))
        repeated["symmetric_bands"]["all"] = {"per_example": {"matched_minus_random": [.1]}}
        with self.assertRaises(ValueError):
            paired_structure_difference(ordinary, repeated)

    def test_structure_is_deterministic_and_shared_band_draws_agree(self):
        a, b = result([0, .4, -.3, .1]), result([.1, .2, .3, .4])
        one = paired_structure_difference(a, b, seed=2)
        two = paired_structure_difference(a, b, seed=2)
        self.assertEqual(one, two)
        self.assertEqual(one["bands"]["early"], one["bands"]["late"])
        json.dumps(one, allow_nan=False)

    def test_correlations_use_log_predictors_and_pairwise_missingness(self):
        rows = [
            {"model_a": "a", "model_b": "b", "late_gap": 1., "smaller_size_m": 10, "size_ratio": 1, "sink_share_a": .1, "sink_share_b": .1},
            {"model_a": "a", "model_b": "c", "late_gap": 2., "smaller_size_m": 100, "size_ratio": None, "sink_share_a": .2, "sink_share_b": .2},
            {"model_a": "b", "model_b": "c", "late_gap": 3., "smaller_size_m": 1000, "size_ratio": 10, "sink_share_a": .3, "sink_share_b": None},
            {"model_a": "a", "model_b": "d", "late_gap": None, "smaller_size_m": 10000, "size_ratio": 100},
        ]
        summary = summarise(rows, ["a", "b", "c", "d", "e"])
        corr = summary["correlations"]
        self.assertEqual(corr["log10_smaller_size_vs_late_gap"]["n_pairs"], 3)
        self.assertAlmostEqual(corr["log10_smaller_size_vs_late_gap"]["pearson_r"], 1)
        self.assertEqual(corr["log10_size_ratio_vs_late_gap"]["n_pairs"], 2)
        self.assertEqual(corr["mean_sink_share_vs_late_gap"]["n_pairs"], 2)
        model_a = summary["per_model"][0]
        self.assertEqual(model_a["n_partners"], 3)
        self.assertEqual(model_a["n_late_gap_pairs"], 2)
        self.assertEqual(model_a["mean_late_gap"], 1.5)
        self.assertIsNone(summary["per_model"][-1]["mean_late_gap"])
        json.dumps(summary, allow_nan=False)

    def test_constant_and_nonpositive_predictors_do_not_produce_fake_correlations(self):
        rows = [{"model_a": "a", "model_b": "b", "late_gap": i, "size_ratio": 1, "smaller_size_m": 0} for i in (1, 2, 3)]
        corr = summarise(rows, ["a", "b"])["correlations"]
        self.assertIsNone(corr["log10_size_ratio_vs_late_gap"]["pearson_r"])
        self.assertEqual(corr["log10_smaller_size_vs_late_gap"]["n_pairs"], 0)

    def test_figures_handle_no_scored_pairs(self):
        rows = [{"coverage": {"status": "deferred_insufficient_aligned_examples"}}]
        with tempfile.TemporaryDirectory() as directory:
            for plot in (plot_depth, plot_factors):
                target = Path(directory) / (plot.__name__ + ".png")
                plot(target, rows)
                self.assertTrue(target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))


if __name__ == "__main__":
    unittest.main()
