"""Comparisons between two models."""

import random
import hashlib
import json
import numpy as np

from .metrics import lower_triangle, cosine


def head_similarity(attn_a, attn_b, layer_a, layer_b, head_a, head_b,
                    drop_col0=True):
    """Mean cosine between two specific heads, averaged over the sentences.
    Sentences whose two models tokenise differently produce matrices of
    different sizes and are skipped.
    """
    scores = []
    for sent_a, sent_b in zip(attn_a, attn_b):
        va = lower_triangle(sent_a[layer_a][head_a], drop_col0)
        vb = lower_triangle(sent_b[layer_b][head_b], drop_col0)
        if len(va) != len(vb) or len(va) == 0:
            continue
        scores.append(cosine(va, vb))
    return float(np.mean(scores)) if scores else 0.0


def head_grid(attn_a, attn_b, layer_a, layer_b, heads_a, heads_b,
              drop_col0=True):
    """Similarity between every head of one layer and every head of the other."""
    grid = np.zeros((len(heads_a), len(heads_b)))
    for ia, ha in enumerate(heads_a):
        for ib, hb in enumerate(heads_b):
            grid[ia, ib] = head_similarity(attn_a, attn_b, layer_a, layer_b,
                                           ha, hb, drop_col0)
    return grid


def best_match(attn_a, attn_b, layer_a, layer_b, heads_a, heads_b,
               drop_col0=True):
    """For each head in A, its best match anywhere in B, averaged."""
    if not heads_a or not heads_b:
        return np.nan
    grid = head_grid(attn_a, attn_b, layer_a, layer_b, heads_a, heads_b,
                     drop_col0)
    return float(grid.max(axis=1).mean())


def random_baseline(attn_a, attn_b, n_layers_a, n_heads_a,
                    n_layers_b, n_heads_b, sink_a, sink_b,
                    cutoff=0.9, n_samples=150, drop_col0=True):
    """Mean similarity between randomly drawn head pairs."""
    scores, tries = [], 0
    while len(scores) < n_samples and tries < n_samples * 12:
        tries += 1
        la, ha = random.randrange(n_layers_a), random.randrange(n_heads_a)
        lb, hb = random.randrange(n_layers_b), random.randrange(n_heads_b)
        if sink_a[la, ha] > cutoff or sink_b[lb, hb] > cutoff:
            continue
        scores.append(head_similarity(attn_a, attn_b, la, lb, ha, hb,
                                      drop_col0))
    return float(np.mean(scores)) if scores else np.nan


def live_heads(sink, layer, n_heads, cutoff=0.9):
    """Heads in a layer whose sink fraction falls under the cutoff."""
    return [h for h in range(n_heads) if sink[layer, h] <= cutoff]


VERSION = 1
METRIC = "mean_cosine_causal_without_first_content_column"
EPS = 1e-12
METRICS = (
    "matched_mean", "random_mean", "matched_minus_random",
    "shuffled_matched_mean", "shuffled_random_mean",
    "shuffled_matched_minus_random", "same_minus_shuffled",
    "gap_same_minus_shuffled",
)


def _attention(value, name):
    x = np.asarray(value)
    if x.ndim != 5 or x.shape[-1] != x.shape[-2]:
        raise ValueError(f"{name} must have shape [N,L,H,T,T]")
    if min(x.shape[:3]) < 1 or x.shape[-1] < 3:
        raise ValueError(f"{name} needs examples, layers, heads, and at least 3 tokens")
    if not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
        raise ValueError(f"{name} must contain real numeric attention weights")
    # Check a layer at a time so validating a memory map does not allocate an
    # additional whole-model boolean array.
    for layer in range(x.shape[1]):
        if not np.isfinite(x[:, layer]).all():
            raise ValueError(f"{name} contains non-finite attention weights")
    return x


def _inputs(a, b):
    a, b = _attention(a, "attn_a"), _attention(b, "attn_b")
    if a.shape[0] != b.shape[0] or a.shape[-1] != b.shape[-1]:
        raise ValueError("Models must have the same example and token counts")
    return a, b


def _sinks(value, shape, name):
    s = np.asarray(value, dtype=float)
    if s.shape != shape or not np.isfinite(s).all():
        raise ValueError(f"{name} must be a finite array of shape {shape}")
    if np.any(s < 0) or np.any(s > 1 + 1e-6):
        raise ValueError(f"{name} must contain fractions between zero and one")
    return s


def _vectors(attention, layer):
    """Return [N,H,usable_cells] unit vectors and number of zero vectors."""
    i, j = np.tril_indices(attention.shape[-1])
    keep = j > 0
    values = np.asarray(attention[:, layer, :, i[keep], j[keep]], dtype=np.float32)
    # Advanced indexing above places the indexed cell dimension first.
    values = np.moveaxis(values, 0, -1)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    good = norms > EPS
    normalized = np.divide(values, norms, out=np.zeros_like(values), where=good)
    return normalized, int(np.count_nonzero(~good))


def fit_matches(attn_a, attn_b, sink_a, sink_b, cutoff=0.9, seed=0):
    """Select best and random heads on the selection passages.

    Each source head chooses its best target at the nearest relative depth.
    Random targets use the same eligible pool; both mappings are fixed afterward.
    Attention has shape [passage, layer, head, query, key]."""
    a, b = _inputs(attn_a, attn_b)
    if not np.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError("cutoff must be between zero and one")
    sa = _sinks(sink_a, a.shape[1:3], "sink_a")
    sb = _sinks(sink_b, b.shape[1:3], "sink_b")
    rng = np.random.default_rng(seed)
    directions = []
    for source, target, x, y, sx, sy in (
        ("a", "b", a, b, sa, sb), ("b", "a", b, a, sb, sa)
    ):
        rows = []
        target_depth = np.linspace(0, 1, y.shape[1])
        for lx, depth in enumerate(np.linspace(0, 1, x.shape[1])):
            ly = int(np.argmin(np.abs(target_depth - depth)))
            hx = np.flatnonzero(sx[lx] <= cutoff)
            hy = np.flatnonzero(sy[ly] <= cutoff)
            row = {
                "source_layer": lx, "target_layer": ly, "depth": float(depth),
                "source_heads": hx.tolist(), "eligible_target_heads": hy.tolist(),
                "matched_heads": [], "random_heads": [],
                "selection_matched_mean": None, "selection_random_mean": None,
                "selection_zero_vectors": {"source": 0, "target": 0},
            }
            if len(hx) and len(hy):
                vx, _ = _vectors(x, lx)
                vy, _ = _vectors(y, ly)
                vx, vy = vx[:, hx], vy[:, hy]
                scores = np.einsum("nhv,nkv->hk", vx, vy, optimize=True) / len(x)
                best = scores.argmax(axis=1)
                random = rng.integers(len(hy), size=len(hx))
                row.update(
                    matched_heads=hy[best].tolist(), random_heads=hy[random].tolist(),
                    selection_matched_mean=float(scores[np.arange(len(hx)), best].mean()),
                    selection_random_mean=float(scores[np.arange(len(hx)), random].mean()),
                    selection_zero_vectors={
                        "source": int((np.linalg.norm(vx, axis=-1) <= EPS).sum()),
                        "target": int((np.linalg.norm(vy, axis=-1) <= EPS).sum()),
                    },
                )
            rows.append(row)
        directions.append({"source": source, "target": target, "rows": rows})
    return {
        "version": VERSION, "metric": METRIC, "token_count": int(a.shape[-1]),
        "selection_examples": int(len(a)), "cutoff": float(cutoff), "seed": int(seed),
        "shapes": {"a": list(a.shape[1:3]), "b": list(b.shape[1:3])},
        "directions": directions,
    }


def _derangements(n, count, rng):
    indices = np.arange(n)
    out = np.empty((count, n), dtype=np.int64)
    for i in range(count):
        permutation = rng.permutation(n)
        while np.any(permutation == indices):
            permutation = rng.permutation(n)
        out[i] = permutation
    return out


def _values(matched, random):
    n = len(matched)
    same_m, same_r = np.diag(matched).astype(float), np.diag(random).astype(float)
    other_m = (matched.sum(axis=1, dtype=float) - same_m) / (n - 1)
    other_r = (random.sum(axis=1, dtype=float) - same_r) / (n - 1)
    return {
        "matched_mean": same_m, "random_mean": same_r,
        "matched_minus_random": same_m - same_r,
        "shuffled_matched_mean": other_m, "shuffled_random_mean": other_r,
        "shuffled_matched_minus_random": other_m - other_r,
        "same_minus_shuffled": same_m - other_m,
        "gap_same_minus_shuffled": (same_m - same_r) - (other_m - other_r),
    }


def _bootstrap_values(matched, random, weights):
    """Bootstrap a paired-input mean and its two-input U-statistic control.

    weights[b,i] is the multiplicity of input i in replicate b. Subtract the
    diagonal once per sampled slot; different slots may contain duplicate
    original observations, as in an ordinary paired nonparametric bootstrap.
    """
    n = len(matched)
    sm = weights @ np.diag(matched) / n
    sr = weights @ np.diag(random) / n
    om = (np.einsum("bi,bi->b", weights @ matched, weights) - n * sm) / (n * (n - 1))
    or_ = (np.einsum("bi,bi->b", weights @ random, weights) - n * sr) / (n * (n - 1))
    return {
        "matched_mean": sm, "random_mean": sr, "matched_minus_random": sm - sr,
        "shuffled_matched_mean": om, "shuffled_random_mean": or_,
        "shuffled_matched_minus_random": om - or_,
        "same_minus_shuffled": sm - om,
        "gap_same_minus_shuffled": (sm - sr) - (om - or_),
    }


def _summary(matched, random, permutations, weights=None):
    if matched is None:
        return {
            **dict.fromkeys(METRICS), "per_example": dict.fromkeys(METRICS),
            "confidence_intervals": None, "derangement_reference": None,
        }
    values = _values(matched, random)
    result = {key: float(value.mean()) for key, value in values.items()}
    result["per_example"] = {key: value.tolist() for key, value in values.items()}
    result["confidence_intervals"] = None
    if weights is not None and len(weights):
        bootstrap = _bootstrap_values(matched, random, weights)
        result["confidence_intervals"] = {
            key: np.quantile(value, [0.025, 0.975]).tolist()
            for key, value in bootstrap.items()
        }
    result["derangement_reference"] = None
    if len(permutations):
        source_indices = np.arange(len(matched))[None, :]
        scores_m = matched[source_indices, permutations].mean(axis=1)
        scores_r = random[source_indices, permutations].mean(axis=1)
        result["derangement_reference"] = {
            "n": len(permutations), "interpretation": "conditional descriptive reference; not a p-value",
            "matched_mean": float(scores_m.mean()),
            "matched_mean_interval_95": np.quantile(scores_m, [0.025, 0.975]).tolist(),
            "matched_minus_random_mean": float((scores_m - scores_r).mean()),
            "matched_minus_random_interval_95": np.quantile(scores_m - scores_r, [0.025, 0.975]).tolist(),
        }
    return result


def _band_results(matrices, permutations, weights):
    output, combined = {}, {}
    for name, predicate in (
        ("early", lambda d: d <= .25), ("late", lambda d: d >= .75),
        ("all", lambda d: True),
    ):
        selected = [pair for depth, pair in matrices if predicate(depth) and pair is not None]
        pair = tuple(np.mean([x[k] for x in selected], axis=0) for k in (0, 1)) if selected else None
        combined[name] = pair
        output[name] = {
            "n_layer_pairs": len(selected),
            **_summary(*(pair if pair is not None else (None, None)), permutations, weights),
        }
    return output, combined


def _contrast(combined, permutations, weights):
    early, late = combined["early"], combined["late"]
    pair = (early[0] - late[0], early[1] - late[1]) if early is not None and late is not None else (None, None)
    return _summary(*pair, permutations, weights)


def evaluate_matches(plan, attn_a, attn_b, seed=1, n_shuffles=100, n_bootstrap=1000):
    """Evaluate fixed head pairs on held-out passages.

    The different-input control averages all off-diagonal passage pairings.
    Shuffles provide a supplementary reference; passage bootstraps recompute
    the control. Early and late bands use source depth <= .25 and >= .75."""
    a, b = _inputs(attn_a, attn_b)
    if len(a) < 2:
        raise ValueError("Different-input evaluation requires at least two test examples")
    if plan.get("version") != VERSION or plan.get("metric") != METRIC:
        raise ValueError("Unrecognised frozen matching plan")
    if plan.get("token_count") != a.shape[-1]:
        raise ValueError("Test token count differs from the matching plan")
    for key, value in (("a", a), ("b", b)):
        if list(value.shape[1:3]) != plan.get("shapes", {}).get(key):
            raise ValueError(f"Test model {key} layer/head dimensions differ from plan")
    if not isinstance(n_shuffles, (int, np.integer)) or n_shuffles < 0:
        raise ValueError("n_shuffles must be a nonnegative integer")
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 0:
        raise ValueError("n_bootstrap must be a nonnegative integer")
    rng_shuffle, rng_boot = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    n = len(a)
    permutations = _derangements(n, n_shuffles, rng_shuffle)
    weights = rng_boot.multinomial(n, np.full(n, 1 / n), size=n_bootstrap).astype(float)
    arrays = {"a": a, "b": b}
    directions, directional_combined = [], []
    for direction in plan["directions"]:
        x, y = arrays[direction["source"]], arrays[direction["target"]]
        rows, matrices = [], []
        for row in direction["rows"]:
            hx, hm, hr = row["source_heads"], row["matched_heads"], row["random_heads"]
            pair = None
            zero = {"source": 0, "matched_target": 0, "random_target": 0}
            if hx and hm:
                if len(hx) != len(hm) or len(hx) != len(hr):
                    raise ValueError("Frozen source, matched and random head lists must have equal lengths")
                vx, _ = _vectors(x, row["source_layer"])
                vy, _ = _vectors(y, row["target_layer"])
                source, matched, random = vx[:, hx], vy[:, hm], vy[:, hr]
                zero = {
                    "source": int((np.linalg.norm(source, axis=-1) <= EPS).sum()),
                    "matched_target": int((np.linalg.norm(matched, axis=-1) <= EPS).sum()),
                    "random_target": int((np.linalg.norm(random, axis=-1) <= EPS).sum()),
                }
                source = source.reshape(n, -1)
                pair = (
                    (source @ matched.reshape(n, -1).T) / len(hx),
                    (source @ random.reshape(n, -1).T) / len(hx),
                )
            rows.append({
                "source_layer": row["source_layer"], "target_layer": row["target_layer"],
                "depth": row["depth"], "n_source_heads": len(hx),
                "n_target_candidates": len(row["eligible_target_heads"]),
                "zero_vectors": zero,
                **_summary(*(pair if pair is not None else (None, None)), permutations),
            })
            matrices.append((row["depth"], pair))
        bands, combined = _band_results(matrices, permutations, weights)
        directional_combined.append(combined)
        directions.append({
            "source": direction["source"], "target": direction["target"],
            "rows": rows, "bands": bands,
            "early_minus_late": _contrast(combined, permutations, weights),
        })
    symmetric_bands, symmetric_combined = {}, {}
    for band in ("early", "late", "all"):
        # Both directions must be available for the symmetric estimate.
        pairs = [d[band] for d in directional_combined]
        pair = tuple((pairs[0][k] + pairs[1][k]) / 2 for k in (0, 1)) if len(pairs) == 2 and all(p is not None for p in pairs) else None
        symmetric_combined[band] = pair
        symmetric_bands[band] = _summary(*(pair if pair is not None else (None, None)), permutations, weights)
    return {
        "version": VERSION, "n_examples": n, "token_count": int(a.shape[-1]),
        "seed": int(seed), "n_shuffles": int(n_shuffles), "n_bootstrap": int(n_bootstrap),
        "control_description": "Exact other-input mean for frozen matches at equal token count; input-associated correspondence, not evidence of shared semantics or reasoning.",
        "bootstrap_description": "Paired example bootstrap with the other-input mean recomputed per replicate; conditional on frozen matching and these model checkpoints.",
        "directions": directions, "symmetric_bands": symmetric_bands,
        "symmetric_early_minus_late": _contrast(symmetric_combined, permutations, weights),
    }


def stable_seed(seed, *names):
    """Keep a pair's random draws independent of the order of comparisons."""
    value = json.dumps([seed, *names], sort_keys=True, ensure_ascii=False)
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16)
