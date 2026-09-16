"""Cache identity follows the passages, while extraction fingerprints stay intact."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.attnlib.extract import AttentionCache, cache_dir, digest, passages_hash


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.passages = {
            "schema_version": 1, "token_count": 4,
            "records": [{"id": "a", "text": "abcd", "split": "selection", "condition": "ordinary"},
                        {"id": "b", "text": "efgh", "split": "evaluation", "condition": "ordinary"}],
            "source_files": {"selection": {"file": "train.txt"}},
        }

    def test_provenance_changes_do_not_change_the_cache_key(self):
        moved = copy.deepcopy(self.passages)
        moved["source_files"] = {"selection": {"file": "copied-train.txt"}}
        moved["tokenizer"] = {"name_or_path": "a-public-tokenizer"}
        self.assertEqual(passages_hash(self.passages), passages_hash(moved))
        self.assertEqual(cache_dir("cache", self.passages, "gpt2"), cache_dir("cache", moved, "gpt2"))

    def test_text_order_split_and_length_changes_change_the_cache_key(self):
        changes = []
        for key, value in (("text", "changed"), ("split", "evaluation")):
            changed = copy.deepcopy(self.passages)
            changed["records"][0][key] = value
            changes.append(changed)
        changed = copy.deepcopy(self.passages)
        changed["records"].reverse()
        changes.append(changed)
        changed = copy.deepcopy(self.passages)
        changed["token_count"] = 8
        changes.append(changed)
        for changed in changes:
            self.assertNotEqual(passages_hash(self.passages), passages_hash(changed))

    def test_loading_preserves_and_checks_original_extraction_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            target = cache_dir(directory, self.passages, "gpt2")
            target.mkdir(parents=True)
            fingerprint = {"version": 2, "hub": "gpt2", "manifest_sha256": "original-provenance-hash"}
            metadata = {
                "model": "gpt2", "fingerprint": fingerprint,
                "passages_sha256": passages_hash(self.passages),
                "records": {r["id"]: {"text_sha256": digest(r["text"]), "n_tokens": 4,
                                         "offsets": [[i, i + 1] for i in range(4)]}
                            for r in self.passages["records"]},
            }
            (target / "metadata.json").write_text(json.dumps(metadata))
            hidden = np.arange(6, dtype=np.float32).reshape(2, 3)
            np.savez(target / "a.npz", hidden=hidden, fingerprint=np.array(digest(fingerprint)))
            cache = AttentionCache(self.passages, directory, ["gpt2"])
            np.testing.assert_array_equal(cache.load_hidden("gpt2", ["a"]), hidden[None])
            np.savez(target / "a.npz", hidden=hidden, fingerprint=np.array("wrong-model-fingerprint"))
            with self.assertRaisesRegex(ValueError, "Stale per-example cache"):
                cache.load_hidden("gpt2", ["a"])
            metadata["passages_sha256"] = "wrong-passage-data"
            (target / "metadata.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "Cache/passages mismatch"):
                AttentionCache(self.passages, directory, ["gpt2"])
