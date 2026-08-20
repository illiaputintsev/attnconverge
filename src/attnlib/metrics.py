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