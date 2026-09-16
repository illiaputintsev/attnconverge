"""Representation geometry on paired examples, with a finite-sample control.

Linear centered kernel alignment (CKA) follows Kornblith et al. (2019):
https://proceedings.mlr.press/v97/kornblith19a.html

Examples, not tokens or neurons, are the observations. Inputs to the main
function are final-content-token hidden states of shape [examples, layers,
features]. Different models may have different layer counts and widths. The
caller must ensure identical examples/order and document extraction semantics.

This is the biased, nonnegative CKA estimator. Its finite-sample permutation
baseline need not be zero. Report that baseline and the excess alongside raw
CKA; excess CKA is neither unbiased CKA nor on the scale of attention cosine.
"""

from __future__ import annotations

import numpy as np


def _unit_gram(features: np.ndarray) -> np.ndarray | None:
    """Compute a centered linear Gram matrix with unit Frobenius norm."""
    if np.all(features == features[0]):
        return None
    # Rescaling before squaring avoids overflow/underflow and preserves CKA.
    scale = np.max(np.abs(features))
    centered = features / scale
    centered -= centered.mean(axis=0, keepdims=True)
    gram = centered @ centered.T
    return _center_and_normalise(gram)


def _center_and_normalise(gram: np.ndarray) -> np.ndarray | None:
    """Center a Gram matrix, then normalise stably."""
    if np.all(gram == gram[0, 0]):
        return None
    centered = gram - gram.mean(axis=0, keepdims=True)
    centered -= centered.mean(axis=1, keepdims=True)
    norm = float(np.sqrt(np.einsum("ij,ij->", centered, centered)))
    if norm == 0 or not np.isfinite(norm):
        return None
    return centered / norm


def _cka(gram_a: np.ndarray, gram_b: np.ndarray) -> float:
    # The exact value lies in [0, 1] for positive semidefinite linear Grams.
    return float(np.clip(np.einsum("ij,ij->", gram_a, gram_b), 0.0, 1.0))


def _null_mean(gram_a: np.ndarray, gram_b: np.ndarray) -> float:
    """Exact CKA expectation under a uniform permutation of example labels.

    For centered K,L, sums of off-diagonal entries equal -tr(K),-tr(L).
    Averaging tr(K P L P.T) over all permutations therefore gives
    tr(K)*tr(L)/(n-1). Both Grams already have Frobenius norm one. This
    expectation includes identity/fixed-point permutations, as does the null
    sampler; it is not a derangement control.
    """
    return float(np.clip(np.trace(gram_a) * np.trace(gram_b)
                         / (len(gram_a) - 1), 0.0, 1.0))


def linear_cka(features_a: np.ndarray, features_b: np.ndarray) -> float | None:
    """Return linear CKA for paired [N,D] features; constant inputs give None."""
    a, b = np.asarray(features_a, dtype=np.float64), np.asarray(features_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or len(a) != len(b):
        raise ValueError("Features must be [examples, features] with equal example counts.")
    if len(a) < 2 or a.shape[1] == 0 or b.shape[1] == 0:
        raise ValueError("At least two examples and one feature are required.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Features must contain only finite values.")
    ka, kb = _unit_gram(a), _unit_gram(b)
    return None if ka is None or kb is None else _cka(ka, kb)


def _depths(n_layers: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_layers) if n_layers > 1 else np.array([0.0])


def _layer_rows(n_a: int, n_b: int) -> list[dict]:
    depths_a, depths_b = _depths(n_a), _depths(n_b)
    rows = []
    for direction, source_depths, target_depths in (
        ("a_to_b", depths_a, depths_b), ("b_to_a", depths_b, depths_a)
    ):
        for source_layer, depth in enumerate(source_depths):
            # np.argmin breaks exact ties toward the lower-index layer.
            target_layer = int(np.argmin(np.abs(target_depths - depth)))
            la, lb = ((source_layer, target_layer) if direction == "a_to_b"
                      else (target_layer, source_layer))
            rows.append({
                "direction": direction, "source_layer": source_layer,
                "target_layer": target_layer, "layer_a": la, "layer_b": lb,
                "depth": float(depth), "depth_a": float(depths_a[la]),
                "depth_b": float(depths_b[lb]),
            })
    return rows


def _interval(values: np.ndarray) -> list[float] | None:
    values = values[np.isfinite(values)]
    return np.quantile(values, [0.025, 0.975]).tolist() if len(values) else None


def _score_record(raw: float, null: float, permuted: np.ndarray) -> dict:
    quantiles = _interval(permuted)
    valid_perm = permuted[np.isfinite(permuted)]
    return {
        "status": "ok" if np.isfinite(raw) else "undefined_no_variation",
        "raw_cka": float(raw) if np.isfinite(raw) else None,
        "shuffled_null_mean": float(null) if np.isfinite(null) else None,
        "shuffled_null_mc_mean": float(valid_perm.mean()) if len(valid_perm) else None,
        "shuffled_null_q025": quantiles[0] if quantiles else None,
        "shuffled_null_q975": quantiles[1] if quantiles else None,
        "excess_cka": float(raw - null) if np.isfinite(raw + null) else None,
        "permutation_pvalue": (
            float((1 + np.count_nonzero(valid_perm >= raw - 1e-12)) / (len(valid_perm) + 1))
            if len(valid_perm) and np.isfinite(raw) else None
        ),
        "raw_cka_ci95": None,
        "excess_cka_ci95": None,
        "n_valid_permutations": int(len(valid_perm)),
        "n_valid_bootstrap": 0,
    }


def _summarise(rows: list[dict], selector, raw: np.ndarray, null: np.ndarray,
               permuted: np.ndarray) -> dict:
    selected = [i for i, row in enumerate(rows) if selector(row["depth"])]
    valid = [i for i in selected if np.isfinite(raw[i])]
    if not valid:
        report = _score_record(np.nan, np.nan, np.array([]))
        report.update(n_rows=len(selected), n_valid_rows=0, n_directions=0)
        return report
    directions = sorted({rows[i]["direction"] for i in valid})
    weights = np.array([
        1.0 / len(directions) / sum(rows[j]["direction"] == rows[i]["direction"]
                                  for j in valid) for i in valid
    ])
    # Use the same example permutation in every layer. Do not treat layer
    # pairs as independent observations or average null quantile endpoints.
    report = _score_record(
        float(raw[valid] @ weights), float(null[valid] @ weights),
        permuted[:, valid] @ weights,
    )
    report.update(n_rows=len(selected), n_valid_rows=len(valid), n_directions=len(directions))
    if len(valid) != len(selected):
        report["status"] = "partial_no_variation"
    return report


def compare_representations(hidden_a: np.ndarray, hidden_b: np.ndarray,
                            seed: int = 0, n_permutations: int = 100,
                            n_bootstrap: int = 0) -> dict:
    """Compare paired final-token states [N,L,D] with fixed relative-depth pairs.

    Every source layer is mapped to its nearest relative-depth target layer in
    BOTH directions. CKA itself is symmetric; the two directions address
    unequal depth grids. Early (<= .25) and late (>= .75) summaries weight
    directions equally, then source layers equally within each direction.

    Permutations break correspondence between whole examples; both Gram axes
    are permuted together. The reported 2.5/97.5 percentiles describe the null
    distribution, not a confidence interval around observed CKA. Permutation
    p-values are one-sided, unadjusted for multiple comparisons and conditional
    on exchangeable examples; correlated excerpts weaken that assumption.

    n_permutations may be zero; the exact permutation mean remains available.
    n_bootstrap is accepted for runner compatibility but no bootstrap CI is
    computed. Ordinary paired bootstrap duplicates become shared structure in
    CKA; shuffling those duplicated rows destroys that structure, producing
    severely misleading excess-CKA intervals even for independent features.
    The requested count is recorded explicitly; absent CIs are JSON nulls.
    Constant layers also produce nulls, never artificial zero similarities.
    """
    a, b = np.asarray(hidden_a, dtype=np.float64), np.asarray(hidden_b, dtype=np.float64)
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0]:
        raise ValueError("Hidden states must be [examples, layers, features] with equal example counts.")
    if len(a) < 4 or min(a.shape[1:]) < 1 or min(b.shape[1:]) < 1:
        raise ValueError("At least four examples, one layer and one feature are required.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Hidden states must contain only finite values.")
    for name, count in (("n_permutations", n_permutations), ("n_bootstrap", n_bootstrap)):
        if not isinstance(count, (int, np.integer)) or isinstance(count, bool) or count < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")

    n = len(a)
    grams_a = [_unit_gram(a[:, layer]) for layer in range(a.shape[1])]
    grams_b = [_unit_gram(b[:, layer]) for layer in range(b.shape[1])]
    rows = _layer_rows(a.shape[1], b.shape[1])
    unique_pairs = sorted({(r["layer_a"], r["layer_b"]) for r in rows})
    pair_index = {pair: i for i, pair in enumerate(unique_pairs)}
    row_indices = [pair_index[(r["layer_a"], r["layer_b"])] for r in rows]
    raw = np.full(len(unique_pairs), np.nan)
    null = raw.copy()
    permuted = np.full((n_permutations, len(unique_pairs)), np.nan)
    valid_pairs = [(i, la, lb) for i, (la, lb) in enumerate(unique_pairs)
                   if grams_a[la] is not None and grams_b[lb] is not None]
    for i, la, lb in valid_pairs:
        raw[i] = _cka(grams_a[la], grams_b[lb])
        null[i] = _null_mean(grams_a[la], grams_b[lb])

    perm_rng = np.random.default_rng(seed)
    for p in range(n_permutations):
        indices = perm_rng.permutation(n)
        shuffled_b = [g[np.ix_(indices, indices)] if g is not None else None for g in grams_b]
        for i, la, lb in valid_pairs:
            permuted[p, i] = _cka(grams_a[la], shuffled_b[lb])

    raw, null = raw[row_indices], null[row_indices]
    permuted = permuted[:, row_indices]
    for i, row in enumerate(rows):
        row.update(_score_record(raw[i], null[i], permuted[:, i]))
    summaries = {
        name: _summarise(rows, select, raw, null, permuted)
        for name, select in (("early", lambda d: d <= .25), ("late", lambda d: d >= .75))
    }
    return {
        "method": "linear_centered_cka_with_example_permutation_control",
        "reference": "https://proceedings.mlr.press/v97/kornblith19a.html",
        "n_examples": n, "n_layers_a": a.shape[1], "n_layers_b": b.shape[1],
        "n_features_a": a.shape[2], "n_features_b": b.shape[2],
        "seed": int(seed), "n_permutations": int(n_permutations),
        "n_bootstrap": 0, "n_bootstrap_requested": int(n_bootstrap),
        "bootstrap_status": "not_computed_naive_paired_bootstrap_is_unreliable_for_biased_cka",
        "rows": rows, "summaries": summaries,
        "notes": [
            "Examples must be identical and in the same order; rows do not validate text identity.",
            "Each feature vector is one example's final-content-token hidden state, not pooled token states.",
            "Layer pairs are fixed by i/(L-1), using both directions; a one-layer stack has depth 0.",
            "shuffled_null_mean is the exact uniform-permutation expectation; quantiles are Monte Carlo.",
            "Raw CKA has finite-sample bias; excess is raw minus that expectation, not unbiased CKA.",
            "Null quantiles are not confidence intervals around the observed CKA score.",
            "Bootstrap CIs are omitted: duplicated example structure makes naive paired bootstrap misleading.",
            "Permutation p-values are one-sided, unadjusted and assume exchangeable examples; source clustering matters.",
            "Invalid constant layers are counted and yield nulls, not zero scores.",
            "Attention cosine and CKA have different scales; their numerical values are not directly comparable.",
        ],
    }
