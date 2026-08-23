"""E5: Agreement against scale, four Pythia sizes

Every pair of models is compared head by head and the gap over the
random-pair baseline is recorded at each relative depth. Six pairs in total,
from 70m to 1b.
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
import random
import os

from src.attnlib import style
from src.attnlib.extract import load_sentences, get_data
from src.attnlib.metrics import sink_fraction, relative_depth_pairs
from src.attnlib.matching import best_match, random_baseline, live_heads

MODELS = [
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
]

SIZES = {"pythia-70m": 70, "pythia-160m": 160, "pythia-410m": 410, "pythia-1b": 1000}

SENTENCES_PATH = "data/sentences.txt"
OUT = "results"
CACHE = "results/attn_cache_pythia4.pkl"

SINK_CUTOFF = 0.9 
N_RANDOM = 150
SEED = 0


def compare(nameA, nameB, data, sinks):
    """One pair: gap over baseline at each relative depth"""
    attnA, LA, HA, _ = data[nameA]
    attnB, LB, HB, _ = data[nameB]
    sinkA, sinkB = sinks[nameA], sinks[nameB]
    ratio = SIZES[nameB] / SIZES[nameA]

    print(f"\n--- {nameA} vs {nameB} ({ratio:.1f}x apart) ---")

    pairs = relative_depth_pairs(LA, LB)
    rows = []
    for (lA, lB, depth) in pairs:
        liveA = live_heads(sinkA, lA, HA, SINK_CUTOFF)
        liveB = live_heads(sinkB, lB, HB, SINK_CUTOFF)
        best = best_match(attnA, attnB, lA, lB, liveA, liveB)
        rand = random_baseline(attnA, attnB, LA, HA, LB, HB,
                                sinkA, sinkB, SINK_CUTOFF, N_RANDOM)
        rows.append({"depth": depth, "best": best, "rand": rand,
                    "nA": len(liveA), "nB": len(liveB)})
        print(f"depth {depth:.2f}  L{lA}/L{lB}  heads {len(liveA)}x{len(liveB)}"
                f"  best {best:.3f}  random {rand:.3f}  gap {best - rand:+.3f}")

    # every layer against every layer
    grid = np.zeros((LA, LB))
    for lA in range(LA):
        liveA = live_heads(sinkA, lA, HA, SINK_CUTOFF)
        for lB in range(LB):
            liveB = live_heads(sinkB, lB, HB, SINK_CUTOFF)
            grid[lA, lB] = best_match(attnA, attnB, lA, lB, liveA, liveB)

    argmax = grid.argmax(axis=1)
    predicted = {a: b for a, b, _ in pairs}
    agree = sum(1 for a in range(LA) if predicted[a] == int(argmax[a]))
    distinct = len(set(int(x) for x in argmax))
    print(f"relative depth correct for {agree}/{LA} layers")
    print(f"{LA} layers of {nameA} map onto {distinct} distinct layers of {nameB}")

    gaps = [r["best"] - r["rand"] for r in rows if not np.isnan(r["best"])]

    return {"rows": rows, "grid": grid, "nameA": nameA, "nameB": nameB,
            "LA": LA, "LB": LB, "ratio": ratio, "agree": agree,
            "early": float(np.mean(gaps[:2])), "mean": float(np.mean(gaps)),
            "argmax": argmax, "predicted": predicted}


def main():
    os.makedirs(OUT, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    style.apply()

    sentences = load_sentences(SENTENCES_PATH)
    print(f"{len(sentences)} sentences, {len(MODELS)} models")

    data = get_data(MODELS, sentences, CACHE)
    names = [m.split("/")[-1] for m in MODELS]

    toks = [data[n][3] for n in names]
    print(f"\nall {len(names)} tokenise identically: {all(t == toks[0] for t in toks)}")

    sinks = {n: sink_fraction(*data[n][:3]) for n in names}
    print(f"\nheads under the {SINK_CUTOFF} sink cutoff:")
    for n in names:
        s = sinks[n]
        print(f"{n}: {(s <= SINK_CUTOFF).sum()} of {s.size}")

    results = [compare(a, b, data, sinks) for a, b in itertools.combinations(names, 2)]

    # how close the two models are in size?
    print("\n=== agreement against size ===")
    ordered = sorted(results, key=lambda r: r["ratio"])
    print(f"{'pair':<32} {'ratio':>7} {'early':>8} {'mean':>8} {'rel-depth':>10}")
    for r in ordered:
        pair = f"{r['nameA']} vs {r['nameB']}"
        print(f"{pair:<32} {r['ratio']:6.1f}x {r['early']:+8.3f} {r['mean']:+8.3f} "
              f"{r['agree']:>6}/{r['LA']}")

    ratios = np.array([r["ratio"] for r in ordered])
    earlies = np.array([r["early"] for r in ordered])
    corr = float(np.corrcoef(np.log10(ratios), earlies)[0, 1])
    print(f"\ncorrelation between size ratio and early gap: {corr:+.3f}")

    # two pairs at similar ratios but different sizes
    tight = ordered[:3]
    smallest = min(tight, key=lambda r: SIZES[r["nameA"]])
    largest = max(tight, key=lambda r: SIZES[r["nameA"]])
    print(f"\n{smallest['nameA']} vs {smallest['nameB']} "
            f"({smallest['ratio']:.1f}x): {smallest['early']:+.3f}")
    print(f"{largest['nameA']} vs {largest['nameB']} "
            f"({largest['ratio']:.1f}x): {largest['early']:+.3f}")
    print(f"difference: {largest['early'] - smallest['early']:+.3f}")

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

    ax = axes[0]
    for k, r in enumerate(results):
        d = [x["depth"] for x in r["rows"]]
        g = [x["best"] - x["rand"] for x in r["rows"]]
        ax.plot(d, g, marker="o", ms=4, color=style.SERIES[k % len(style.SERIES)],
                label=f"{r['nameA']} vs {r['nameB']}")
    ax.axhline(0, color=style.MUTED, lw=1, ls="--")
    ax.set_xlabel("relative depth")
    ax.set_ylabel("gap over random baseline")
    style.titled(ax, f"All {len(results)} pairs",
            f"{len(sentences)} sentences, sink cutoff {SINK_CUTOFF}")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter(ratios, earlies, s=70, color=style.SERIES[0], zorder=3)
    for r in ordered:
        ax.annotate(f"{r['nameA'].replace('pythia-', '')}/"
                    f"{r['nameB'].replace('pythia-', '')}",
                    (r["ratio"], r["early"]), fontsize=8,
                    xytext=(5, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("size ratio between the pair")
    ax.set_ylabel("early-layer gap over baseline")
    style.titled(ax, "Agreement against size ratio", f"r = {corr:+.2f}")

    plt.tight_layout()
    plt.savefig(f"{OUT}/e5_scale.png", dpi=150)
    plt.close()

    # layer grids
    ncols = 3
    nrows = (len(results) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.6 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for k, r in enumerate(results):
        ax = axes[k]
        im = ax.imshow(r["grid"], cmap=style.HEATMAP, aspect="auto",
                        vmin=0.2, vmax=0.9)
        ax.set_xlabel(f"{r['nameB']} layer")
        ax.set_ylabel(f"{r['nameA']} layer")
        ax.set_title(f"{r['nameA']} vs {r['nameB']} ({r['ratio']:.1f}x)", fontsize=10)
        ax.grid(False)
        for lA in range(r["LA"]):
            ax.plot(r["predicted"][lA], lA, marker="s", ms=6,
                    mfc="none", mec="white", mew=1.2)
            ax.plot(int(r["argmax"][lA]), lA, marker="o", ms=3.5, color="red")
        fig.colorbar(im, ax=ax)

    for k in range(len(results), len(axes)):
        axes[k].axis("off")

    fig.text(0.5, 0.005, "white square = relative-depth prediction, "
             "red dot = actual best match",
             ha="center", fontsize=9, color=style.MUTED)
    plt.tight_layout(rect=[0, 0.025, 1, 1])
    plt.savefig(f"{OUT}/e5_layer_grids.png", dpi=150)
    plt.close()

    print(f"\nsaved plots to {OUT}/")
    print(f"attention cache at {CACHE} (delete it to re-extract)")


if __name__ == "__main__":
    main()