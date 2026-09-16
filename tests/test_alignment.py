import unittest

import numpy as np

from src.attnlib.metrics import aggregate_spans, common_span_ends, exact_alignment
from src.attnlib.metrics import conditional_content


class AlignmentTests(unittest.TestCase):
    def test_equal_counts_do_not_align(self):
        a = {"n_tokens": 3, "offsets": [[0, 2], [2, 3], [3, 5]]}
        b = {"n_tokens": 3, "offsets": [[0, 1], [1, 3], [3, 5]]}
        self.assertFalse(exact_alignment([a, b], "abcde", 3))
        self.assertTrue(exact_alignment([a, a], "abcde", 3))

    def test_unicode_overlapping_offsets_rejected(self):
        a = {"n_tokens": 3, "offsets": [[0, 1], [0, 1], [1, 2]]}
        self.assertFalse(exact_alignment([a, a], "ab", 3))

    def test_shared_spans_preserve_query_prefix_and_key_mass(self):
        offsets_a = [[i, i + 1] for i in range(8)]
        offsets_b = [[0, 2], [2, 4], [4, 6], [6, 8]]
        ends = common_span_ends([offsets_a, offsets_b], 8, 4)
        self.assertEqual(ends, [2, 4, 6, 8])
        a = np.tril(np.ones((8, 8)))
        a /= a.sum(axis=-1, keepdims=True)
        reduced = aggregate_spans(a, offsets_a, ends)
        expected = np.tril(np.ones((4, 4)))
        expected /= expected.sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(reduced, expected)
        np.testing.assert_allclose(reduced.sum(-1), 1)
        np.testing.assert_allclose(np.triu(reduced, 1), 0)

    def test_bos_mass_not_renormalized_before_storage(self):
        content = np.array([[0.4, 0], [0.1, 0.1]])
        np.testing.assert_allclose(conditional_content(content), [[1, 0], [.5, .5]])
        np.testing.assert_allclose(content.sum(-1), [.4, .2])


if __name__ == "__main__":
    unittest.main()
