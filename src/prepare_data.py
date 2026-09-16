"""Prepare disjoint selection and evaluation passages, with paired repetition."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata


SELECTION_SOURCE = "data/wikitext_train.txt"
EVALUATION_SOURCE = "data/wikitext_test.txt"
SELECTION = "data/selection.txt"
EVALUATION = "data/evaluation.txt"
REPETITION = "data/repetition.txt"
TOKENIZER = "EleutherAI/pythia-70m"
TOKENIZER_REVISION = "a39f36b100fe8a5377810d56c3f4789b9c53ac42"
N_SELECTION = 100
N_EVALUATION = 100
N_TOKENS = 32
SEED = 20260911


def _text_hash(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFC", text).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _read_source(path: str | Path) -> tuple[dict, list[dict], dict]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Source must be an existing plain-text file: {path}")
    try:
        raw = path.read_bytes()
        content = raw.decode("utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read source as UTF-8 text: {path}: {exc}") from exc
    stats = {"input_lines": len(content.splitlines()), "blank_lines": 0,
             "heading_lines": 0, "duplicate_sources": 0}
    seen: set[str] = set()
    rows = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        text = line.strip()
        if not text:
            stats["blank_lines"] += 1
            continue
        if re.fullmatch(r"=+\s*.*?\s*=+", text):
            stats["heading_lines"] += 1
            continue
        source_id = _text_hash(text)
        if source_id in seen:
            stats["duplicate_sources"] += 1
            continue
        seen.add(source_id)
        rows.append({"source_id": source_id, "source_line": line_number,
                     "source_text": text})
    if not rows:
        raise ValueError(f"Source contains no nonempty, non-heading text rows: {path}")
    return {"file": path.name, "sha256": hashlib.sha256(raw).hexdigest()}, rows, stats


def _encode(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def exact_token_prefix(text: str, tokenizer, token_count: int) -> str | None:
    """Cut at a character boundary and verify the requested token count.

    Re-encoding handles prefix-sensitive merges and shared byte-token offsets."""
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    if len(ids) < token_count:
        return None
    if len(ids) != len(offsets):
        raise ValueError("Tokenizer returned inconsistent IDs and offset mapping lengths")
    boundaries = set()
    # Nearby boundaries cover shared Unicode offsets and prefix-sensitive merges
    # without repeatedly encoding every prefix of a long source paragraph.
    for start, end in offsets[max(0, token_count - 3):token_count + 3]:
        if not (0 <= start <= end <= len(text)):
            raise ValueError("Tokenizer offsets must index characters in the source text")
        boundaries.update((start, end))
    expected_end = offsets[token_count - 1][1]
    for end in sorted(boundaries, key=lambda value: (abs(value - expected_end), value)):
        prefix = text[:end]
        if prefix and len(_encode(tokenizer, prefix)) == token_count:
            return prefix
    return None


def _repeated_variant(ordinary: str, tokenizer, token_count: int) -> str | None:
    unit = exact_token_prefix(ordinary, tokenizer, 8)
    if unit is None:
        return None
    # Keep the exact raw prefix; add a separator only when it has no trailing
    # whitespace. The repeated text is tokenized afresh because boundary tokens
    # can differ from their sentence-initial forms.
    piece = unit + ("" if unit[-1].isspace() else " ")
    repeats = max(4, token_count // 8 + 2)
    for _ in range(8):
        text = exact_token_prefix(piece * repeats, tokenizer, token_count)
        if text is not None:
            return text if _text_hash(text) != _text_hash(ordinary) else None
        repeats *= 2
    return None


def _prepare_rows(rows: list[dict], tokenizer, tokens: int, evaluation: bool,
                  stats: dict) -> list[dict]:
    stats.update({"no_exact_window": 0, "duplicate_windows": 0,
                  "duplicate_repetition_windows": 0,
                  "no_distinct_repetition": 0})
    seen = set()
    seen_repetitions = set()
    prepared = []
    for row in rows:
        text = exact_token_prefix(row["source_text"], tokenizer, tokens)
        if text is None:
            stats["no_exact_window"] += 1
            continue
        text_id = _text_hash(text)
        if text_id in seen:
            stats["duplicate_windows"] += 1
            continue
        repetition = _repeated_variant(text, tokenizer, tokens) if evaluation else None
        if evaluation and repetition is None:
            stats["no_distinct_repetition"] += 1
            continue
        if evaluation and _text_hash(repetition) in seen_repetitions:
            stats["duplicate_repetition_windows"] += 1
            continue
        seen.add(text_id)
        if evaluation:
            seen_repetitions.add(_text_hash(repetition))
        prepared.append({"source_id": row["source_id"], "source_line": row["source_line"],
                         "text": text, "text_id": text_id, "repetition": repetition})
    return prepared


def prepare_passages(selection_path, evaluation_path, tokenizer,
                     n_selection: int = 100, n_evaluation: int = 100,
                     tokens: int = 32, seed: int = 0,
                     tokenizer_name: str | None = None,
                     tokenizer_revision: str | None = None) -> dict:
    """Sample ordinary passages and paired repetition variants.

    Duplicate sources and windows shared across splits are excluded.
    Token lengths are checked after slicing the original text."""
    for name, value in (("n_selection", n_selection), ("n_evaluation", n_evaluation)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 8:
        raise ValueError("tokens must be an integer greater than 8 for an eight-token repetition unit")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer with character offset mappings is required")

    sel_info, selection, sel_stats = _read_source(selection_path)
    eval_info, evaluation, eval_stats = _read_source(evaluation_path)
    shared_sources = {row["source_id"] for row in selection} & {
        row["source_id"] for row in evaluation}
    selection = [row for row in selection if row["source_id"] not in shared_sources]
    evaluation = [row for row in evaluation if row["source_id"] not in shared_sources]
    sel_stats["cross_split_sources"] = eval_stats["cross_split_sources"] = len(shared_sources)
    selection = _prepare_rows(selection, tokenizer, tokens, False, sel_stats)
    evaluation = _prepare_rows(evaluation, tokenizer, tokens, True, eval_stats)

    sel_windows = {row["text_id"] for row in selection}
    eval_windows = {value for row in evaluation
                    for value in (row["text_id"], _text_hash(row["repetition"]))}
    shared_windows = sel_windows & eval_windows
    kept_selection = [row for row in selection if row["text_id"] not in shared_windows]
    kept_evaluation = [row for row in evaluation
                       if row["text_id"] not in shared_windows
                       and _text_hash(row["repetition"]) not in shared_windows]
    sel_stats["cross_split_windows"] = len(selection) - len(kept_selection)
    eval_stats["cross_split_windows"] = len(evaluation) - len(kept_evaluation)
    selection, evaluation = kept_selection, kept_evaluation
    sel_stats["eligible_rows"] = len(selection)
    eval_stats["eligible_rows"] = len(evaluation)

    for split, rows, requested, stats in (("selection", selection, n_selection, sel_stats),
                                         ("evaluation", evaluation, n_evaluation, eval_stats)):
        if len(rows) < requested:
            raise ValueError(f"Insufficient eligible {split} source rows: requested {requested}, "
                             f"found {len(rows)} after filtering. Details: {json.dumps(stats, sort_keys=True)}")
    rng = random.Random(seed)
    rng.shuffle(selection)
    rng.shuffle(evaluation)
    records = []
    for split, rows, count in (("selection", selection, n_selection),
                               ("evaluation", evaluation, n_evaluation)):
        for row in rows[:count]:
            group_id = f"{split}-{row['source_id']}"
            common = {"split": split, "source_id": row["source_id"],
                      "source_line": row["source_line"],
                      "paired_id": group_id if split == "evaluation" else None}
            records.append({**common, "id": f"{group_id}-ordinary", "condition": "ordinary",
                            "text": row["text"]})
            if split == "evaluation":
                records.append({**common, "id": f"{group_id}-repetition", "condition": "repetition",
                                "text": row["repetition"]})
    name = tokenizer_name or str(getattr(tokenizer, "name_or_path", ""))
    if Path(name).expanduser().is_absolute() or Path(name).expanduser().is_dir():
        name = Path(name).name
    tokenizer_info = {"name_or_path": name, "class": type(tokenizer).__name__,
                      "is_fast": True, "add_special_tokens": False}
    if tokenizer_revision:
        tokenizer_info["revision"] = tokenizer_revision
    return {
        "schema_version": 1, "token_count": tokens, "seed": seed,
        "source_files": {"selection": sel_info, "evaluation": eval_info},
        "tokenizer": tokenizer_info,
        "counts": {"selection": n_selection, "evaluation_pairs": n_evaluation,
                   "evaluation_records": 2 * n_evaluation},
        "preparation": {"source_unit": "nonempty non-heading line", "source_line_base": 1,
                        "hash_normalization": "NFC, casefold, collapsed whitespace",
                        "repetition_unit_tokens": 8,
                        "repetition_interpretation": "changes repetition and lexical diversity",
                        "selection": sel_stats, "evaluation": eval_stats},
        "records": records,
    }


def write_passages(passages, selection_path, evaluation_path, repetition_path):
    """Save the three text sets, keeping evaluation pairs in matching line order."""
    for path, split, condition in ((selection_path, "selection", "ordinary"),
                                    (evaluation_path, "evaluation", "ordinary"),
                                    (repetition_path, "evaluation", "repetition")):
        lines = [r["text"] for r in passages["records"]
                 if r["split"] == split and r["condition"] == condition]
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION, use_fast=True)
    passages = prepare_passages(SELECTION_SOURCE, EVALUATION_SOURCE, tokenizer,
                                 N_SELECTION, N_EVALUATION, N_TOKENS, SEED)
    write_passages(passages, SELECTION, EVALUATION, REPETITION)
    print(f"{N_SELECTION} selection excerpts: {SELECTION}")
    print(f"{N_EVALUATION} evaluation excerpts: {EVALUATION}")
    print(f"{N_EVALUATION} repetition variants: {REPETITION}")


if __name__ == "__main__":
    main()
