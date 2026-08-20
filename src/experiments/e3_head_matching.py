"""E3: Cross-model head matching, 70m against 160m

For each pair of layers matched by relative depth, every head in model A is
compared against every head in model B by cosine similarity over their
attention matrices, averaged across sentences. For each head in A, the best match in B is kept.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import os

from src.attnlib.extract import load_sentences, get_data
from src.attnlib.metrics import sink_fraction, relative_depth_pairs
from src.attnlib.matching import best_match, random_baseline, live_heads

MODEL_A = "EleutherAI/pythia-70m"
MODEL_B = "EleutherAI/pythia-160m"
SENTENCES_PATH = "data/sentences.txt"
OUT = "results"
CACHE = "results/attn_cache.pkl"

SINK_CUTOFF = 0.7  # mean sink fraction threshold
N_RANDOM = 200  # random head pairs sampled per layer pair
SEED = 0


def main():
    os.makedirs(OUT, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)

    sentences = load_sentences(SENTENCES_PATH)
    print(f"{len(sentences)} sentences")

    data = get_data([MODEL_A, MODEL_B], sentences, CACHE)
    attnA, LA, HA, _ = data[MODEL_A.split("/")[-1]]
    attnB, LB, HB, _ = data[MODEL_B.split("/")[-1]]

    sinkA = sink_fraction(attnA, LA, HA)
    sinkB = sink_fraction(attnB, LB, HB)

    print(f"\nheads surviving the {SINK_CUTOFF} sink cutoff:")
    print(f"{MODEL_A.split('/')[-1]}: {(sinkA <= SINK_CUTOFF).sum()} of {LA * HA}")
    print(f"{MODEL_B.split('/')[-1]}: {(sinkB <= SINK_CUTOFF).sum()} of {LB * HB}")

    # match layers by relative depth: for each layer of A, the nearest of B
    pairs = relative_depth_pairs(LA, LB)

    results = {"masked": [], "full": []}

    for mode, drop in [("masked", True), ("full", False)]:
        print(f"\n=== {mode} (column 0 {'removed' if drop else 'kept'}) ===")
        for (lA, lB, depth) in pairs:
            liveA = live_heads(sinkA, lA, HA, SINK_CUTOFF)
            liveB = live_heads(sinkB, lB, HB, SINK_CUTOFF)

            # best match per head in A, and a floor of random head pairs
            best = best_match(attnA, attnB, lA, lB, liveA, liveB, drop)
            rand = random_baseline(attnA, attnB, LA, HA, LB, HB,
                                   sinkA, sinkB, SINK_CUTOFF, N_RANDOM, drop)

            results[mode].append({
                "depth": depth, "lA": lA, "lB": lB,
                "best": best, "rand": rand,
                "nA": len(liveA), "nB": len(liveB),
            })

            if np.isnan(best):
                print(f"depth {depth:.2f}  L{lA}/L{lB} "
                      f"no live heads (A:{len(liveA)} B:{len(liveB)})")
            else:
                print(f"depth {depth:.2f}  L{lA}/L{lB} "
                      f"heads {len(liveA)}x{len(liveB)} "
                      f"best {best:.3f}  random {rand:.3f} "
                      f"gap {best - rand:+.3f}")

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, mode in zip(axes, ["masked", "full"]):
        r = results[mode]
        d = [x["depth"] for x in r]
        best = [x["best"] for x in r]
        base = [x["rand"] for x in r]
        ax.plot(d, best, marker="o", color="#1d9e75", label="best match per head")
        ax.plot(d, base, marker="s", ls="--", color="#888", label="random head pairs")
        ax.fill_between(d, base, best, where=[not np.isnan(x) for x in best],
                        color="#1d9e75", alpha=0.12)
        ax.set_xlabel("Relative depth")
        ax.set_title("column 0 removed" if mode == "masked" else "full matrix")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Mean cosine similarity")
    plt.suptitle(f"Cross-model head matching, {len(sentences)} sentences, "
                 f"sink cutoff {SINK_CUTOFF}")
    plt.tight_layout()
    plt.savefig(f"{OUT}/e3_head_matching.png", dpi=150)
    plt.close()

    # summary
    print("\nSummary:")
    for mode in ["masked", "full"]:
        gaps = [x["best"] - x["rand"] for x in results[mode]
                if not np.isnan(x["best"]) and not np.isnan(x["rand"])]
        if gaps:
            print(f"{mode}: mean gap {np.mean(gaps):+.3f}, "
                  f"largest {max(gaps):+.3f}, smallest {min(gaps):+.3f}")

    print(f"\nsaved plot to {OUT}/e3_head_matching.png")
    print(f"attention cache at {CACHE} (delete it to re-extract)")


if __name__ == "__main__":
    main()