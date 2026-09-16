"""Measurements taken on attention matrices."""

import numpy as np


def lower_triangle(m, drop_col0=True):
    """Flatten the usable cells of one attention matrix into a vector."""
    seq = m.shape[0]
    start = 1 if drop_col0 else 0
    vals = []
    for i in range(1, seq):
        for j in range(start, i + 1):
            vals.append(m[i, j])
    return np.array(vals, dtype=np.float32)


def cosine(a, b):
    """Cosine similarity between two vectors, zero if either is degenerate."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def sink_fraction(attn, n_layers, n_heads):
    """Mean weight on token 0 for every (layer, head), averaged over sentences.
    Row 0 is excluded. Returns an array of shape [n_layers, n_heads].
    """
    acc = np.zeros((n_layers, n_heads))
    for sent in attn:
        for l in range(n_layers):
            acc[l] += sent[l][:, 1:, 0].astype(np.float32).mean(axis=1)
    return acc / len(attn)


def row_entropy(m, normalise=True):
    """Entropy of each row, optionally normalised.

    Measures how spread out a row is. 
    Row 0 is skipped when normalising, since ln(1) is zero.
    """
    seq = m.shape[0]
    p = np.clip(m.astype(np.float32), 1e-12, None)   # log(0) is -inf
    h = -(p * np.log(p)).sum(axis=-1)
    if not normalise:
        return h
    return np.array([h[i] / np.log(i + 1) for i in range(1, seq)],
                    dtype=np.float32)


def band(n_layers, lo, hi):
    """Layer indices inside a fractional band of the stack."""
    a = min(int(round(n_layers * lo)), n_layers - 1)
    b = max(a + 1, int(round(n_layers * hi)))
    return list(range(a, min(b, n_layers)))


def relative_depth_pairs(n_layers_a, n_layers_b):
    """Match each layer of one model to the nearest layer of the other.
    Returns a list of (layer_a, layer_b, depth).
    """
    depths_a = [i / (n_layers_a - 1) for i in range(n_layers_a)]
    depths_b = [i / (n_layers_b - 1) for i in range(n_layers_b)]
    out = []
    for i, da in enumerate(depths_a):
        j = int(np.argmin([abs(da - db) for db in depths_b]))
        out.append((i, j, da))
    return out


def conditional_content(attention):
    a = np.asarray(attention, dtype=np.float32)
    mass = a.sum(axis=-1, keepdims=True)
    return np.divide(a, mass, out=np.zeros_like(a), where=mass > 0)


def valid_offsets(offsets, text_length):
    """Require a non-overlapping complete partition of the original text."""
    if not offsets or offsets[0][0] != 0 or offsets[-1][1] != text_length:
        return False
    return all(a < b and (i == 0 or offsets[i - 1][1] == a)
               for i, (a, b) in enumerate(offsets))


def exact_alignment(records, text, token_count):
    offsets = [r["offsets"] for r in records]
    return (all(r["n_tokens"] == token_count for r in records)
            and all(valid_offsets(x, len(text)) for x in offsets)
            and all(x == offsets[0] for x in offsets[1:]))


def common_span_ends(offset_sets, text_length, n_spans=16):
    """Choose span ends at equal rank intervals among shared token boundaries."""
    if n_spans < 4 or not offset_sets:
        raise ValueError("At least four spans and one tokenization are required")
    if not all(valid_offsets(x, text_length) for x in offset_sets):
        return None
    common = sorted(set.intersection(*(set(b for _, b in x) for x in offset_sets)))
    if len(common) < n_spans:
        return None
    positions = np.ceil(np.arange(1, n_spans + 1) * len(common) / n_spans).astype(int) - 1
    return [common[i] for i in positions]


def aggregate_spans(attention, offsets, span_ends):
    """Use each span's final query and sum its key weights.

    Final-token queries keep the visible text prefix identical across models."""
    a = np.asarray(attention, dtype=np.float32)
    native_ends = [b for _, b in offsets]
    if a.shape[-2:] != (len(offsets), len(offsets)):
        raise ValueError("Attention and token offsets disagree")
    if not span_ends or span_ends[-1] != native_ends[-1] or any(x not in native_ends for x in span_ends):
        raise ValueError("Span boundaries must end at complete native tokens and cover the passage")
    if any(x >= y for x, y in zip(span_ends, span_ends[1:])):
        raise ValueError("Span boundaries must be strictly increasing")
    end_indices = [native_ends.index(x) for x in span_ends]
    queries = a[..., end_indices, :]
    starts = [0] + [i + 1 for i in end_indices[:-1]]
    return np.stack([queries[..., start:end + 1].sum(axis=-1)
                     for start, end in zip(starts, end_indices)], axis=-1)


STRUCTURE_METRICS = ("matched_minus_random", "same_minus_shuffled", "gap_same_minus_shuffled")


def paired_structure_difference(ordinary_result, repetition_result, seed=0, n_bootstrap=1000):
    """Repetition minus ordinary scores on paired passages.

    Bootstrap intervals apply to the matched-minus-random gap. The input-excess
    changes use different reference pools and are returned as point estimates."""
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 0:
        raise ValueError("n_bootstrap must be a nonnegative integer")
    if ordinary_result.get("n_examples") != repetition_result.get("n_examples"):
        raise ValueError("Structure conditions must contain the same paired example count")
    rng = np.random.default_rng(seed)
    draw_cache = {}
    bands = {}
    for band in ("early", "late", "all"):
        original = ordinary_result.get("symmetric_bands", {}).get(band, {}).get("per_example", {})
        repeated = repetition_result.get("symmetric_bands", {}).get(band, {}).get("per_example", {})
        output = {"n_examples": int(ordinary_result.get("n_examples", 0))}
        for metric in STRUCTURE_METRICS:
            x, y = original.get(metric), repeated.get(metric)
            estimate = {"mean_change": None, "confidence_interval": None, "n_pairs": 0}
            if x is not None and y is not None:
                x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
                if x.ndim != 1 or x.shape != y.shape or len(x) != output["n_examples"]:
                    raise ValueError(f"Unaligned per-example values for {band}/{metric}")
                valid = np.isfinite(x) & np.isfinite(y)
                difference = y[valid] - x[valid]
                if len(difference):
                    estimate["mean_change"] = float(difference.mean())
                    estimate["n_pairs"] = int(len(difference))
                    if metric == "matched_minus_random" and len(difference) >= 2 and n_bootstrap:
                        # The same draws are reused across bands when their
                        # paired sample masks agree.
                        cache_key = tuple(np.flatnonzero(valid))
                        if cache_key not in draw_cache:
                            draw_cache[cache_key] = rng.integers(len(difference), size=(n_bootstrap, len(difference)))
                        sampled = difference[draw_cache[cache_key]].mean(axis=1)
                        estimate["confidence_interval"] = np.quantile(sampled, [.025, .975]).tolist()
            output[metric] = estimate
        bands[band] = output
    return {
        "direction": "repetition_minus_ordinary", "seed": int(seed),
        "n_bootstrap": int(n_bootstrap), "bands": bands,
        "notes": [
            "Example IDs/order and the same frozen head mappings must be verified by the caller.",
            "Gap intervals use paired example bootstrap, conditional on the selected heads and model checkpoints.",
            "Input-excess changes are point estimates only: their condition-specific other-input reference pools cannot be recomputed from these per-example summaries.",
        ],
    }


def sink_summary(data, cutoff):
    start = data["sink_start"].mean(axis=0)
    return {"parked_share": float((start > cutoff).mean()),
            "mean_bos_mass": float(data["sink_bos"].mean()),
            "mean_first_region_mass": float(data["sink_first_region"].mean()),
            "mean_native_first_content_mass": float(data["sink_native_first_content"].mean()),
            "by_layer_parked_share": (start > cutoff).mean(axis=1).tolist()}
