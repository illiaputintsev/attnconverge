"""E6: Independently trained models

Experiment E3 repeated on a pair from outside the Pythia family. Models
GPT-2 and GPT-Neo share a tokeniser, but have different organisations,
corpora (WebText and the Pile), separate training runs, and at 124M
and 125M almost the same size and shape.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import os
from matplotlib.lines import Line2D
from src.attnlib import style
from src.attnlib.extract import load_sentences, get_data
from src.attnlib.metrics import sink_fraction, relative_depth_pairs
from src.attnlib.matching import best_match, random_baseline, live_heads

MODEL_A = "gpt2" # OpenAI, WebText, 124M
MODEL_B = "EleutherAI/gpt-neo-125m"  # EleutherAI, The Pile, 125M

SENTENCES_PATH = "data/sentences.txt"
OUT = "results"
CACHE = "results/attn_cache_gpt.pkl"

SINK_CUTOFF = 0.9
N_RANDOM = 150
SEED = 0


def attention_types(name, n_layers):
    """Which layers use local attention, for models that alternate.
    """
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(name)
    types = getattr(cfg, "attention_layers", None)
    return list(types) if types else ["global"] * n_layers


def main():
    os.makedirs(OUT, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    style.apply()

    sentences = load_sentences(SENTENCES_PATH)
    print(f"{len(sentences)} sentences")

    data = get_data([MODEL_A, MODEL_B], sentences, CACHE)
    nameA, nameB = MODEL_A.split("/")[-1], MODEL_B.split("/")[-1]
    attnA, LA, HA, tokA = data[nameA]
    attnB, LB, HB, tokB = data[nameB]

    print(f"\nidentical tokenisation: {tokA == tokB}")
    if tokA != tokB:
        print(f"{nameA}: {tokA}")
        print(f"{nameB}: {tokB}")
        print("Matrices are not comparable cell by cell")
        return

    typesB = attention_types(MODEL_B, LB)
    if set(typesB) != {"global"}:
        print(f"{nameB} attention types: {typesB}")

    sinkA = sink_fraction(attnA, LA, HA)
    sinkB = sink_fraction(attnB, LB, HB)
    print(f"\nheads under the {SINK_CUTOFF} sink cutoff:")
    print(f"{nameA}: {(sinkA <= SINK_CUTOFF).sum()} of {sinkA.size}")
    print(f"{nameB}: {(sinkB <= SINK_CUTOFF).sum()} of {sinkB.size}")

    print(f"\n=== {nameA} vs {nameB}, by relative depth ===")
    pairs = relative_depth_pairs(LA, LB)
    rows = []
    for (lA, lB, depth) in pairs:
        liveA = live_heads(sinkA, lA, HA, SINK_CUTOFF)
        liveB = live_heads(sinkB, lB, HB, SINK_CUTOFF)
        best = best_match(attnA, attnB, lA, lB, liveA, liveB)
        rand = random_baseline(attnA, attnB, LA, HA, LB, HB,
                                sinkA, sinkB, SINK_CUTOFF, N_RANDOM)
        kind = typesB[lB] if lB < len(typesB) else "global"
        rows.append({"depth": depth, "best": best, "rand": rand,
                    "nA": len(liveA), "nB": len(liveB), "typeB": kind})
        print(f"depth {depth:.2f}  L{lA}/L{lB}  heads {len(liveA)}x{len(liveB)}  "
                f"best {best:.3f}  random {rand:.3f}  gap {best - rand:+.3f}   "
                f"{nameB} L{lB} is {kind}")

    if set(typesB) != {"global"}:
        print(f"\n=== paired against global and local layers ===")
        for kind in ["global", "local"]:
            g = [r["best"] - r["rand"] for r in rows
                if r["typeB"] == kind and not np.isnan(r["best"])]
            if g:
                print(f"{kind}: mean gap {np.mean(g):+.3f} over {len(g)} layer pairs")

    # every layer against every layer
    print("\n=== full layer grid ===")
    grid = np.zeros((LA, LB))
    for lA in range(LA):
        liveA = live_heads(sinkA, lA, HA, SINK_CUTOFF)
        for lB in range(LB):
            liveB = live_heads(sinkB, lB, HB, SINK_CUTOFF)
            grid[lA, lB] = best_match(attnA, attnB, lA, lB, liveA, liveB)

    argmax = grid.argmax(axis=1)
    predicted = {a: b for a, b, _ in pairs}
    for lA in range(LA):
        found = int(argmax[lA])
        tag = "ok" if predicted[lA] == found else f"predicted L{predicted[lA]}"
        print(f"{nameA} L{lA} -> {nameB} L{found}  ({grid[lA, found]:.3f})  {tag}")

    agree = sum(1 for a in range(LA) if predicted[a] == int(argmax[a]))
    distinct = len(set(int(x) for x in argmax))
    print(f"\nrelative depth correct for {agree}/{LA} layers")
    print(f"{LA} layers map onto {distinct} distinct layers")

    gaps = [r["best"] - r["rand"] for r in rows if not np.isnan(r["best"])]
    print(f"\nearly gap {np.mean(gaps[:2]):+.3f}")
    print(f"late gap {np.mean(gaps[-3:]):+.3f}")
    print(f"mean gap {np.mean(gaps):+.3f}")

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

    ax = axes[0]
    d = [r["depth"] for r in rows]
    g = [r["best"] - r["rand"] for r in rows]
    ax.plot(d, g, marker="o", color=style.SERIES[3], label=f"{nameA} vs {nameB}")
    ax.axhline(0, color=style.MUTED, lw=1, ls="--")
    ax.set_xlabel("relative depth")
    ax.set_ylabel("gap over random baseline")
    style.titled(ax, "Models trained independently",
                f"{len(sentences)} sentences, sink cutoff {SINK_CUTOFF}")
    ax.legend()

    ax = axes[1]
    im = ax.imshow(grid, cmap=style.HEATMAP, aspect="auto", vmin=0.2, vmax=0.9)
    ax.set_xlabel(f"{nameB} layer")
    ax.set_ylabel(f"{nameA} layer")
    ax.set_xticks(range(LB))
    ax.set_yticks(range(LA))
    ax.grid(False)
    for lA in range(LA):
        ax.plot(predicted[lA], lA, marker="s", ms=9, mfc="none",
                mec="white", mew=1.6, zorder=3)
        ax.plot(int(argmax[lA]), lA, marker="o", ms=6, color=style.SERIES[3],
                mec="none", ls="none", zorder=4)
    fig.colorbar(im, ax=ax, label="best-match similarity")

    handles = [
        Line2D([], [], marker="s", ms=8, mfc="none", mec="#666", mew=1.4,
               ls="none", label="predicted by relative depth"),
        Line2D([], [], marker="o", ms=6, color=style.SERIES[3], mec="none",
               ls="none", label="best match found"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 0.995),
              ncol=2, fontsize=8, frameon=False)
    ax.set_title("Every layer against every layer", loc="left", pad=24)

    plt.tight_layout()
    plt.savefig(f"{OUT}/e6_independent.png", dpi=150)
    plt.close()

    print(f"\nsaved plot to {OUT}/e6_independent.png")
    print(f"attention cache at {CACHE} (delete it to re-extract)")


if __name__ == "__main__":
    main()
