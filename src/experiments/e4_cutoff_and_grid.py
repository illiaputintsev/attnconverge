"""E4: Filter sensitivity and layer alignment, 70m against 160m

The matching from E3 is repeated at four sink cutoffs (0.7, 0.8, 0.9, and 1.0,
which keeps every head) since E3's late figures came from only two or three
heads. Every layer of A is then compared against every layer of B to check
which layers actually match.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt

from src.attnlib import style
from src.attnlib.extract import load_sentences, get_data
from src.attnlib.metrics import sink_fraction, relative_depth_pairs
from src.attnlib.matching import (best_match, random_baseline, live_heads,
                                  head_similarity)

MODELS = ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m"]
SENTENCES_PATH = "data/sentences.txt"
OUT = "results"
CACHE = "results/attn_cache_e3.pkl"

CUTOFFS = [0.7, 0.8, 0.9, 1.0]
GRID_CUTOFF = 0.9
N_RANDOM = 150
SEED = 0


def main():
    os.makedirs(OUT, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    style.apply()

    sentences = load_sentences(SENTENCES_PATH)
    print(f"{len(sentences)} sentences")

    data = get_data(MODELS, sentences, CACHE)
    short_a, short_b = [m.split("/")[-1] for m in MODELS]
    attn_a, LA, HA, _ = data[short_a]
    attn_b, LB, HB, _ = data[short_b]

    sink_a = sink_fraction(attn_a, LA, HA)
    sink_b = sink_fraction(attn_b, LB, HB)
    pairs = relative_depth_pairs(LA, LB)
    predicted = {a: b for a, b, _ in pairs}

    # sink filter sweep
    print("\n=== sink filter sweep ===")
    sweep = {}
    for cutoff in CUTOFFS:
        label = "no filter" if cutoff >= 1.0 else f"cutoff {cutoff}"
        n_a = int((sink_a <= cutoff).sum())
        n_b = int((sink_b <= cutoff).sum())
        print(f"\n{label}  ({n_a}/{LA * HA} and {n_b}/{LB * HB} heads kept)")

        rows = []
        for layer_a, layer_b, depth in pairs:
            heads_a = live_heads(sink_a, layer_a, HA, cutoff)
            heads_b = live_heads(sink_b, layer_b, HB, cutoff)
            best = best_match(attn_a, attn_b, layer_a, layer_b,
                              heads_a, heads_b)
            rand = random_baseline(attn_a, attn_b, LA, HA, LB, HB,
                                   sink_a, sink_b, cutoff, N_RANDOM)
            rows.append({"depth": depth, "best": best, "rand": rand,
                         "n_a": len(heads_a), "n_b": len(heads_b)})
            print(f"depth {depth:.2f}  L{layer_a}/L{layer_b}  "
                  f"heads {len(heads_a)}x{len(heads_b)}  "
                  f"best {best:.3f}  random {rand:.3f}  "
                  f"gap {best - rand:+.3f}")
        sweep[cutoff] = rows

    unfiltered = sweep[1.0]
    first_gap = unfiltered[0]["best"] - unfiltered[0]["rand"]
    worst_gap = min(r["best"] - r["rand"] for r in unfiltered)
    print(f"\nwith no filter, all {HA} heads are used at every depth")
    print(f"gap at the first layer: {first_gap:+.3f}")
    print(f"lowest gap: {worst_gap:+.3f}")

    tight_floor = np.mean([r["rand"] for r in sweep[0.7]])
    open_floor = np.mean([r["rand"] for r in sweep[1.0]])
    print(f"\nmean baseline at cutoff 0.7: {tight_floor:.3f}")
    print(f"mean baseline with no filter: {open_floor:.3f}")

    # full layer grid
    print("\n=== full layer grid ===")
    grid = np.zeros((LA, LB))
    for layer_a in range(LA):
        heads_a = live_heads(sink_a, layer_a, HA, GRID_CUTOFF)
        for layer_b in range(LB):
            heads_b = live_heads(sink_b, layer_b, HB, GRID_CUTOFF)
            grid[layer_a, layer_b] = best_match(attn_a, attn_b,
                                                layer_a, layer_b,
                                                heads_a, heads_b)
        print(f"{short_a} layer {layer_a} done")

    argmax = grid.argmax(axis=1)
    print("\nbest-matching layer against what relative depth predicts:")
    for layer_a in range(LA):
        found = int(argmax[layer_a])
        tag = "ok" if predicted[layer_a] == found \
            else f"predicted L{predicted[layer_a]}"
        print(f"{short_a} L{layer_a} -> {short_b} L{found}  "
              f"({grid[layer_a, found]:.3f})  {tag}")

    agree = sum(1 for a in range(LA) if predicted[a] == int(argmax[a]))
    distinct = len(set(int(x) for x in argmax))
    print(f"\nrelative depth correct for {agree}/{LA} layers")
    print(f"{LA} layers of {short_a} map onto {distinct} distinct layers "
          f"of {short_b}")

    # within-model baseline
    print("\n=== within-model baseline ===")
    within, tries = [], 0
    while len(within) < N_RANDOM and tries < N_RANDOM * 12:
        tries += 1
        l1, h1 = random.randrange(LA), random.randrange(HA)
        l2, h2 = random.randrange(LA), random.randrange(HA)
        if (l1, h1) == (l2, h2):
            continue
        if sink_a[l1, h1] > GRID_CUTOFF or sink_a[l2, h2] > GRID_CUTOFF:
            continue
        within.append(head_similarity(attn_a, attn_a, l1, l2, h1, h2))

    within_mean = float(np.mean(within))
    cross_mean = float(np.mean([r["rand"] for r in sweep[GRID_CUTOFF]]))
    print(f"two random heads from {short_a}: {within_mean:.3f}")
    print(f"a random {short_a} head against a {short_b} head: {cross_mean:.3f}")
    print(f"difference: {cross_mean - within_mean:+.3f}")

    # plots
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

    ax = axes[0]
    for i, cutoff in enumerate(CUTOFFS):
        rows = sweep[cutoff]
        label = "no filter" if cutoff >= 1.0 else f"cutoff {cutoff}"
        ax.plot([r["depth"] for r in rows],
                [r["best"] - r["rand"] for r in rows],
                marker="o", color=style.SERIES[i % len(style.SERIES)],
                label=label)
    ax.axhline(0, color=style.MUTED, lw=1, ls="--")
    ax.set_xlabel("relative depth")
    ax.set_ylabel("gap over random baseline")
    style.titled(ax, "Gap over baseline at four sink cutoffs",
                 "the shape holds at every setting, including none")
    ax.legend(title="sink filter")

    ax = axes[1]
    im = ax.imshow(grid, cmap=style.HEATMAP, aspect="auto")
    ax.set_xlabel(f"{short_b} layer")
    ax.set_ylabel(f"{short_a} layer")
    ax.set_xticks(range(LB))
    ax.set_yticks(range(LA))
    ax.grid(False)
    for layer_a in range(LA):
        ax.plot(predicted[layer_a], layer_a, marker="s", ms=9,
                mfc="none", mec="white", mew=1.6)
        ax.plot(int(argmax[layer_a]), layer_a, marker="o", ms=5, color="red")
    fig.colorbar(im, ax=ax, label="best-match similarity")
    style.titled(ax, "Every layer against every layer",
                 "white square = relative depth, red dot = actual best match")

    plt.tight_layout()
    plt.savefig(f"{OUT}/e4_cutoff_and_grid.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, cutoff in enumerate(CUTOFFS):
        rows = sweep[cutoff]
        label = "no filter" if cutoff >= 1.0 else f"cutoff {cutoff}"
        ax.plot([r["depth"] for r in rows],
                [min(r["n_a"], r["n_b"]) for r in rows],
                marker="o", color=style.SERIES[i % len(style.SERIES)],
                label=label)
    ax.set_xlabel("relative depth")
    ax.set_ylabel("smaller of the two head counts")
    style.titled(ax, "How many heads each point rests on",
                 "the late points in E3 rested on two")
    ax.legend(title="sink filter")
    plt.tight_layout()
    plt.savefig(f"{OUT}/e4_head_counts.png", dpi=150)
    plt.close()

    print(f"\nsaved plots to {OUT}/")


if __name__ == "__main__":
    main()