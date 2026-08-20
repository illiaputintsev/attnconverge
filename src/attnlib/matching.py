"""Comparisons between two models."""

import random
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