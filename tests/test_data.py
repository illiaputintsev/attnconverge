import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest

from src.attnlib.extract import load_passages, passages_hash
from src.prepare_data import exact_token_prefix, prepare_passages, write_passages


class WordTokenizer:
    """Fast-tokenizer stand-in with real character offsets and no dependencies."""

    is_fast = True
    name_or_path = "test-word-tokenizer"

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        assert not add_special_tokens
        matches = list(re.finditer(r"\S+", text))
        result = {"input_ids": [int(hashlib.sha256(m.group().encode()).hexdigest()[:8], 16)
                                for m in matches]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(m.start(), m.end()) for m in matches]
        return result


class ByteCharacterTokenizer:
    """Simulate byte fallback: an emoji has multiple IDs sharing one offset."""

    is_fast = True

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids, offsets = [], []
        for index, char in enumerate(text):
            encoded = list(char.encode("utf-8"))
            ids.extend(encoded)
            offsets.extend([(index, index + 1)] * len(encoded))
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


def paragraph(prefix, count=40):
    return " ".join(f"{prefix}{i}" for i in range(count))


class PassageDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tokenizer = WordTokenizer()

    def tearDown(self):
        self.temp.cleanup()

    def sources(self, selection, evaluation):
        paths = (self.root / "train.txt", self.root / "test.txt")
        for path, rows in zip(paths, (selection, evaluation)):
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return paths

    def test_manifest_is_deterministic_exact_length_and_paired(self):
        paths = self.sources(["", " = Heading = "] + [paragraph(f"s{i}-") for i in range(8)],
                             [paragraph(f"e{i}-") for i in range(8)])
        first = prepare_passages(*paths, self.tokenizer, 3, 4, seed=17)
        self.assertEqual(first, prepare_passages(*paths, self.tokenizer, 3, 4, seed=17))
        self.assertEqual(len(first["records"]), 11)
        self.assertEqual(first["source_files"]["selection"]["sha256"],
                         hashlib.sha256(paths[0].read_bytes()).hexdigest())
        selection = [r for r in first["records"] if r["split"] == "selection"]
        evaluation = [r for r in first["records"] if r["split"] == "evaluation"]
        self.assertTrue(all(r["source_line"] >= 3 for r in selection))
        self.assertTrue(all(r["paired_id"] is None for r in selection))
        self.assertTrue({r["source_id"] for r in selection}.isdisjoint(
            r["source_id"] for r in evaluation))
        for record in first["records"]:
            self.assertEqual(len(self.tokenizer(record["text"])["input_ids"]), 32)
        for ordinary, repetition in zip(evaluation[::2], evaluation[1::2]):
            self.assertEqual(ordinary["condition"], "ordinary")
            self.assertEqual(repetition["condition"], "repetition")
            self.assertEqual(ordinary["paired_id"], repetition["paired_id"])
            self.assertEqual(ordinary["source_line"], repetition["source_line"])
            self.assertEqual(repetition["text"].split(), ordinary["text"].split()[:8] * 4)
        second = prepare_passages(*paths, self.tokenizer, 3, 4, seed=18)
        self.assertNotEqual([r["id"] for r in first["records"]],
                            [r["id"] for r in second["records"]])

    def test_moving_sources_does_not_change_prepared_data_or_expose_directories(self):
        paths = self.sources([paragraph(f"s{i}-") for i in range(4)],
                             [paragraph(f"e{i}-") for i in range(4)])
        moved = self.root / "another-computer"
        moved.mkdir()
        for path in paths:
            (moved / path.name).write_bytes(path.read_bytes())
        first = prepare_passages(*paths, self.tokenizer, 2, 2, seed=17)
        second = prepare_passages(*(moved / p.name for p in paths), self.tokenizer, 2, 2, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(first["source_files"]["selection"]["file"], "train.txt")
        self.assertNotIn(str(self.root), json.dumps(first))

    def test_local_tokenizer_directory_is_not_saved_as_an_absolute_path(self):
        paths = self.sources([paragraph("s-")], [paragraph("e-")])
        self.tokenizer.name_or_path = str(self.root / "private" / "tokenizer")
        passages = prepare_passages(*paths, self.tokenizer, 1, 1)
        self.assertEqual(passages["tokenizer"]["name_or_path"], "tokenizer")
        self.assertNotIn(str(self.root), json.dumps(passages))
        passages = prepare_passages(*paths, self.tokenizer, 1, 1,
                                     tokenizer_name="example/public-tokenizer", tokenizer_revision="abc123")
        self.assertEqual(passages["tokenizer"]["name_or_path"], "example/public-tokenizer")
        self.assertEqual(passages["tokenizer"]["revision"], "abc123")

    def test_excludes_normalized_source_and_prefix_overlap(self):
        shared = paragraph("SHARED-")
        prefix = paragraph("prefix-", 32)
        paths = self.sources([shared, shared.lower(), prefix + " train-tail", paragraph("safe-s-")],
                             ["  " + shared.lower().replace(" ", "   ") + "  ",
                              prefix + " eval-tail", paragraph("safe-e-")])
        manifest = prepare_passages(*paths, self.tokenizer, 1, 1)
        self.assertTrue(manifest["records"][0]["text"].startswith("safe-s-"))
        self.assertTrue(manifest["records"][1]["text"].startswith("safe-e-"))
        stats = manifest["preparation"]
        self.assertEqual(stats["selection"]["duplicate_sources"], 1)
        for split in ("selection", "evaluation"):
            self.assertEqual(stats[split]["cross_split_sources"], 1)
            self.assertEqual(stats[split]["cross_split_windows"], 1)

    def test_repetition_cannot_duplicate_selection_text(self):
        evaluation_text = paragraph("e-")
        repeated = " ".join(evaluation_text.split()[:8] * 4)
        paths = self.sources([repeated, paragraph("safe-s-")],
                             [evaluation_text, paragraph("safe-e-")])
        manifest = prepare_passages(*paths, self.tokenizer, 1, 1)
        self.assertTrue(manifest["records"][0]["text"].startswith("safe-s-"))
        self.assertTrue(manifest["records"][1]["text"].startswith("safe-e-"))

    def test_deduplicates_windows_and_rejects_short_or_identical_variants(self):
        prefix = paragraph("same-", 32)
        paths = self.sources(["short", prefix + " tail1", prefix + " tail2"],
                             ["repeat " * 40, paragraph("e-")])
        manifest = prepare_passages(*paths, self.tokenizer, 1, 1)
        self.assertEqual(manifest["preparation"]["selection"]["no_exact_window"], 1)
        self.assertEqual(manifest["preparation"]["selection"]["duplicate_windows"], 1)
        self.assertEqual(manifest["preparation"]["evaluation"]["no_distinct_repetition"], 1)
        with self.assertRaisesRegex(ValueError, "requested 2, found 1"):
            prepare_passages(*paths, self.tokenizer, 2, 1)

    def test_identical_repetition_variants_do_not_create_extra_evaluation_pairs(self):
        shared_unit = paragraph("same-", 8)
        paths = self.sources([paragraph("s-")],
                             [shared_unit + " " + paragraph("tail1-"),
                              shared_unit + " " + paragraph("tail2-")])
        manifest = prepare_passages(*paths, self.tokenizer, 1, 1)
        self.assertEqual(manifest["preparation"]["evaluation"]["duplicate_repetition_windows"], 1)
        with self.assertRaisesRegex(ValueError, "Insufficient eligible evaluation"):
            prepare_passages(*paths, self.tokenizer, 1, 2)

    def test_unicode_offsets_do_not_cut_a_multibyte_character(self):
        tokenizer = ByteCharacterTokenizer()
        self.assertIsNone(exact_token_prefix("a🙂bc", tokenizer, 3))
        self.assertEqual(exact_token_prefix("a🙂bc", tokenizer, 5), "a🙂")
        text = paragraph("café🙂-")
        prefix = exact_token_prefix(text, self.tokenizer, 32)
        self.assertEqual(prefix, " ".join(text.split()[:32]))

    def test_source_paths_empty_inputs_and_tokenizer_errors_are_clear(self):
        paths = self.sources([paragraph("s-")], [paragraph("e-")])
        with self.assertRaisesRegex(ValueError, "existing plain-text file"):
            prepare_passages(self.root / "missing.txt", paths[1], self.tokenizer, 1, 1)
        paths[0].write_text("\n = Heading =\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no nonempty, non-heading"):
            prepare_passages(*paths, self.tokenizer, 1, 1)
        with self.assertRaisesRegex(ValueError, "fast tokenizer"):
            prepare_passages(*paths, object(), 1, 1)
        with self.assertRaisesRegex(ValueError, "greater than 8"):
            prepare_passages(*paths, self.tokenizer, 1, 1, tokens=8)


class PlainTextDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.paths = tuple(self.root / name for name in
                           ("selection.txt", "evaluation.txt", "repetition.txt"))

    def tearDown(self):
        self.temp.cleanup()

    def write_inputs(self, selection, evaluation, repetition):
        for path, rows in zip(self.paths, (selection, evaluation, repetition)):
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    def test_text_roundtrip_preserves_unicode_whitespace_and_pairs(self):
        selection = [" café 🙂  ", "A separate selection excerpt."]
        ordinary = ["First held-out excerpt. ", "A second held-out excerpt."]
        repeated = ["First First First. ", "A second A second."]
        self.write_inputs(selection, ordinary, repeated)
        passages = load_passages(*self.paths, tokens=32)
        records = passages["records"]
        self.assertEqual(passages["token_count"], 32)
        self.assertEqual([r["text"] for r in records],
                         selection + [ordinary[0], repeated[0], ordinary[1], repeated[1]])
        self.assertTrue(all(r["split"] == "selection" and r["paired_id"] is None
                            for r in records[:2]))
        for original, repetition in zip(records[2::2], records[3::2]):
            self.assertEqual(original["split"], "evaluation")
            self.assertEqual(repetition["split"], "evaluation")
            self.assertEqual(original["condition"], "ordinary")
            self.assertEqual(repetition["condition"], "repetition")
            self.assertEqual(original["paired_id"], repetition["paired_id"])
        self.assertNotEqual(records[2]["paired_id"], records[4]["paired_id"])

        copied_paths = tuple(self.root / ("copy-" + path.name) for path in self.paths)
        write_passages(passages, *copied_paths)
        for original, copied in zip(self.paths, copied_paths):
            self.assertEqual(original.read_bytes(), copied.read_bytes())
        self.assertEqual(passages, load_passages(*copied_paths))

    def test_missing_partner_or_blank_line_is_rejected(self):
        cases = [
            (["selection"], ["first", "second"], ["first first"]),
            (["selection"], ["first", "", "second"],
             ["first first", "middle middle", "second second"]),
            (["selection"], ["first", "second"], ["first first", "   "]),
        ]
        for rows in cases:
            with self.subTest(rows=rows):
                self.write_inputs(*rows)
                with self.assertRaises(ValueError):
                    load_passages(*self.paths)

    def test_selection_overlap_with_either_evaluation_condition_is_rejected(self):
        for selection in ("ordinary", "repeated repeated"):
            with self.subTest(selection=selection):
                self.write_inputs([selection], ["ordinary"], ["repeated repeated"])
                with self.assertRaises(ValueError):
                    load_passages(*self.paths)

    def test_moving_text_files_preserves_cache_identity(self):
        self.write_inputs(["selection"], ["ordinary"], ["repeated repeated"])
        original = load_passages(*self.paths)
        moved = self.root / "another-directory"
        moved.mkdir()
        copied_paths = tuple(moved / path.name for path in self.paths)
        for source, destination in zip(self.paths, copied_paths):
            destination.write_bytes(source.read_bytes())
        self.assertEqual(passages_hash(original),
                         passages_hash(load_passages(*copied_paths)))


if __name__ == "__main__":
    unittest.main()
