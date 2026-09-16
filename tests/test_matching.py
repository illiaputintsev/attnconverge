"""Methodological checks using causal, row-normalized synthetic attention."""

import copy
import json
import unittest

import numpy as np

from src.attnlib.matching import (
    _bootstrap_values, _derangements, _vectors, evaluate_matches, fit_matches,
)


def attention(patterns, layers=1, tokens=6):
    """patterns[n][h] is 'diagonal', 'previous', or 'first_non_sink'."""
    result = np.zeros((len(patterns), layers, len(patterns[0]), tokens, tokens), dtype=np.float32)
    for n, heads in enumerate(patterns):
        for h, kind in enumerate(heads):
            for q in range(tokens):
                if kind == "diagonal":
                    k = q
                elif kind == "previous":
                    k = max(q - 1, 0)
                elif kind == "first_non_sink":
                    k = min(q, 1)
                else:
                    raise ValueError(kind)
                result[n, :, h, q, k] = 1
    return result


def sinks(x):
    return np.zeros(x.shape[1:3])


def evaluate(plan, a, b, **kwargs):
    return evaluate_matches(plan, a, b, n_shuffles=12, n_bootstrap=50, **kwargs)


class HeldoutMatchingTests(unittest.TestCase):
    def test_vector_domain_and_normalization_match_explicit_definition(self):
        a = attention([["diagonal", "previous"], ["previous", "diagonal"]], layers=3)
        actual, zero = _vectors(a, 2)
        self.assertEqual(actual.shape, (2, 2, 15))
        self.assertEqual(zero, 0)
        expected = np.array([a[1, 2, 0, i, j] for i in range(1, 6) for j in range(1, i + 1)])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(actual[1, 0], expected)

    def test_positional_previous_token_matching_has_no_input_excess(self):
        train = attention([["previous"]] * 3)
        plan = fit_matches(train, train, sinks(train), sinks(train))
        test = attention([["previous"]] * 6)
        result = evaluate(plan, test, test)
        row = result["directions"][0]["rows"][0]
        self.assertAlmostEqual(row["matched_mean"], 1, places=6)
        self.assertAlmostEqual(row["shuffled_matched_mean"], 1, places=6)
        self.assertAlmostEqual(row["same_minus_shuffled"], 0, places=6)
        ci = result["symmetric_bands"]["early"]["confidence_intervals"]["same_minus_shuffled"]
        np.testing.assert_allclose(ci, [0, 0], atol=1e-7)

    def test_input_dependent_correspondence_has_positive_excess(self):
        patterns = [["diagonal"], ["first_non_sink"]] * 3
        train = attention(patterns)
        plan = fit_matches(train, train, sinks(train), sinks(train))
        result = evaluate(plan, attention(patterns), attention(patterns))
        row = result["directions"][0]["rows"][0]
        self.assertAlmostEqual(row["matched_mean"], 1, places=6)
        self.assertGreater(row["same_minus_shuffled"], .4)
        self.assertAlmostEqual(row["same_minus_shuffled"], .48, places=6)

    def test_frozen_matches_are_not_reselected_when_test_head_order_changes(self):
        a = attention([["diagonal"]] * 4)
        b = attention([["diagonal", "previous"]] * 4)
        plan = fit_matches(a, b, sinks(a), sinks(b))
        original = copy.deepcopy(plan)
        self.assertEqual(plan["directions"][0]["rows"][0]["matched_heads"], [0])
        changed = attention([["previous", "diagonal"]] * 4)
        result = evaluate(plan, a, changed)
        self.assertLess(result["directions"][0]["rows"][0]["matched_mean"], .01)
        self.assertEqual(plan, original)

    def test_empty_eligible_heads_are_missing_not_zero(self):
        a = attention([["diagonal"]] * 3, layers=2)
        plan = fit_matches(a, a, np.ones((2, 1)), sinks(a), cutoff=.9)
        result = evaluate(plan, a, a)
        for direction in result["directions"]:
            for row in direction["rows"]:
                self.assertIsNone(row["matched_mean"])
                self.assertIsNone(row["per_example"]["same_minus_shuffled"])
        self.assertIsNone(result["symmetric_bands"]["early"]["same_minus_shuffled"])
        json.dumps(result, allow_nan=False)

    def test_random_targets_use_the_same_frozen_layer_eligible_pool(self):
        a = attention([["diagonal", "previous"]] * 4, layers=2)
        b = attention([["diagonal", "previous", "first_non_sink"]] * 4, layers=3)
        sb = np.array([[1., 0., 1.]] * 3)
        plan = fit_matches(a, b, sinks(a), sb)
        for row in plan["directions"][0]["rows"]:
            self.assertEqual(row["matched_heads"], [1, 1])
            self.assertEqual(row["random_heads"], [1, 1])
            self.assertEqual(row["eligible_target_heads"], [1])
        self.assertEqual([r["target_layer"] for r in plan["directions"][0]["rows"]], [0, 2])
        self.assertEqual([r["target_layer"] for r in plan["directions"][1]["rows"]], [0, 0, 1])
        result = evaluate(plan, a, b)
        self.assertEqual([(d["source"], d["target"]) for d in result["directions"]], [("a", "b"), ("b", "a")])

    def test_outputs_are_deterministic_and_json_serializable(self):
        a = attention([["diagonal", "previous"], ["first_non_sink", "diagonal"]] * 3, layers=2)
        plan = fit_matches(a, a, sinks(a), sinks(a), seed=23)
        self.assertEqual(plan, fit_matches(a, a, sinks(a), sinks(a), seed=23))
        self.assertEqual(evaluate(plan, a, a, seed=24), evaluate(plan, a, a, seed=24))
        json.dumps(evaluate(plan, a, a), allow_nan=False)

    def test_primary_other_input_mean_is_exact_and_independent_of_shuffle_draws(self):
        a = attention([["diagonal"], ["first_non_sink"], ["previous"]])
        plan = fit_matches(a, a, sinks(a), sinks(a))
        result0 = evaluate_matches(plan, a, a, n_shuffles=0, n_bootstrap=0)
        result1 = evaluate(plan, a, a)
        row0, row1 = [r["directions"][0]["rows"][0] for r in (result0, result1)]
        self.assertEqual(row0["shuffled_matched_mean"], row1["shuffled_matched_mean"])
        self.assertIsNone(row0["derangement_reference"])
        vectors, _ = _vectors(a, 0)
        expected = np.mean([np.dot(vectors[i, 0], vectors[j, 0]) for i in range(3) for j in range(3) if i != j])
        self.assertAlmostEqual(row0["shuffled_matched_mean"], expected, places=6)

    def test_relative_depth_bands_and_paired_early_late_contrast(self):
        variable = attention([["diagonal"], ["first_non_sink"]] * 3, layers=5)
        fixed = attention([["previous"]] * 6, layers=5)
        variable[:, 3:] = fixed[:, 3:]
        plan = fit_matches(variable, variable, sinks(variable), sinks(variable))
        result = evaluate(plan, variable, variable)
        bands = result["directions"][0]["bands"]
        self.assertEqual(bands["early"]["n_layer_pairs"], 2)  # depths 0, .25
        self.assertEqual(bands["late"]["n_layer_pairs"], 2)  # depths .75, 1
        self.assertAlmostEqual(bands["early"]["same_minus_shuffled"], .48, places=6)
        self.assertAlmostEqual(bands["late"]["same_minus_shuffled"], 0, places=6)
        contrast = result["symmetric_early_minus_late"]
        self.assertAlmostEqual(contrast["same_minus_shuffled"], .48, places=6)
        self.assertIsNotNone(contrast["confidence_intervals"]["same_minus_shuffled"])

    def test_zero_masked_vectors_are_reported_separately_from_missing_heads(self):
        a = attention([["diagonal"]] * 3)
        a[:] = 0
        a[..., 0] = 1  # all probability is on the masked first content token
        plan = fit_matches(a, a, np.ones((1, 1)), np.ones((1, 1)), cutoff=1)
        result = evaluate(plan, a, a)
        row = result["directions"][0]["rows"][0]
        self.assertEqual(row["zero_vectors"]["source"], 3)
        self.assertEqual(row["zero_vectors"]["matched_target"], 3)
        self.assertEqual(row["matched_mean"], 0)
        self.assertEqual(row["n_source_heads"], 1)

    def test_derangements_have_no_fixed_points(self):
        for n in (2, 3, 20):
            permutations = _derangements(n, 30, np.random.default_rng(0))
            self.assertTrue(np.all(permutations != np.arange(n)))
            for p in permutations:
                np.testing.assert_array_equal(np.sort(p), np.arange(n))

    def test_bootstrap_recomputes_different_input_reference(self):
        m = np.eye(3)
        values = _bootstrap_values(m, np.zeros((3, 3)), np.array([[2., 1., 0.]]))
        self.assertAlmostEqual(values["matched_mean"][0], 1)
        self.assertAlmostEqual(values["shuffled_matched_mean"][0], 1 / 3)
        self.assertAlmostEqual(values["same_minus_shuffled"][0], 2 / 3)

    def test_invalid_shapes_sinks_and_single_test_example_fail(self):
        a = attention([["diagonal"]] * 3)
        with self.assertRaises(ValueError):
            fit_matches(a, a[:2], sinks(a), sinks(a))
        with self.assertRaises(ValueError):
            fit_matches(a, a, np.array([[np.nan]]), sinks(a))
        plan = fit_matches(a, a, sinks(a), sinks(a))
        with self.assertRaises(ValueError):
            evaluate(plan, a[:1], a[:1])
        with self.assertRaises(ValueError):
            evaluate_matches(plan, a, a, n_shuffles=-1)


if __name__ == "__main__":
    unittest.main()
