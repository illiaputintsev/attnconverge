"""E2: attention concentration and sink strength by depth, over many sentences."""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import matplotlib.pyplot as plt
import numpy as np
import math
import os

MODELS = ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m"]
SENTENCES_PATH = "data/sentences.txt"
OUT = "results"


def load_sentences(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def normalised_row_entropy(m):
    """Entropy of each row, divided by that row's own maximum."""
    seq = m.shape[0]
    p = m.clamp_min(1e-12)
    h = -(p * p.log()).sum(dim=-1)# raw entropy per row
    out = []
    for i in range(1, seq):# skip row 0
        ceiling = math.log(i + 1)
        out.append((h[i] / ceiling).item())
    return out


def sink_fraction_rows(m):
    """Weight placed on token 0 by each row, excluding row 0 itself."""
    return m[1:, 0].tolist()


def uniform_sink_baseline(seq):
    """Mean weight on token 0 if every row spread its attention evenly."""
    return float(np.mean([1.0 / (i + 1) for i in range(1, seq)]))


def run_model(name, sentences):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager")
    model.eval()

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    print(f"\n{name}: {n_layers} layers, {n_heads} heads")

    # accumulators: one list per layer, filled across all sentences
    ent = [[] for _ in range(n_layers)]
    sink = [[] for _ in range(n_layers)]
    # per-head sink: [layer][head] -> list of one value per sentence
    head_sink = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
    baselines = []
    lengths = []
    first_tokens = None

    for s_i, text in enumerate(sentences):
        inputs = tok(text, return_tensors="pt")
        seq = inputs["input_ids"].shape[1]
        if seq < 4:
            continue
        lengths.append(seq)
        baselines.append(uniform_sink_baseline(seq))
        if first_tokens is None:
            first_tokens = tok.convert_ids_to_tokens(inputs["input_ids"][0])

        with torch.no_grad():
            out = model(**inputs, output_attentions=True)

        for l, layer in enumerate(out.attentions):
            m = layer[0]
            for h in range(n_heads):
                ent[l].extend(normalised_row_entropy(m[h]))
                rows = sink_fraction_rows(m[h])
                sink[l].extend(rows)
                head_sink[l][h].append(float(np.mean(rows)))

        if (s_i + 1) % 10 == 0:
            print(f"  {s_i + 1}/{len(sentences)} sentences")

    return {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "ent_mean": [float(np.mean(x)) for x in ent],
        "ent_std": [float(np.std(x)) for x in ent],
        "sink_mean": [float(np.mean(x)) for x in sink],
        "sink_std": [float(np.std(x)) for x in sink],
        "head_sink": head_sink,
        "baseline": float(np.mean(baselines)),
        "mean_len": float(np.mean(lengths)),
        "first_tokens": first_tokens,
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    sentences = load_sentences(SENTENCES_PATH)
    print(f"{len(sentences)} sentences loaded")

    results = {}
    for name in MODELS:
        results[name.split("/")[-1]] = run_model(name, sentences)

    keys = list(results.keys())
    same = results[keys[0]]["first_tokens"] == results[keys[1]]["first_tokens"]
    print(f"\nidentical tokenisation on first sentence: {same}")

    #plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colours = {keys[0]: "#1f77b4", keys[1]: "#ff7f0e"}

    for short, r in results.items():
        n = r["n_layers"]
        depth = [i / (n - 1) for i in range(n)]
        c = colours[short]

        em = np.array(r["ent_mean"])
        es = np.array(r["ent_std"])
        ax1.plot(depth, em, marker="o", color=c, label=f"{short} ({n} layers)")
        ax1.fill_between(depth, em - es, em + es, color=c, alpha=0.15)

        sm = np.array(r["sink_mean"])
        ss = np.array(r["sink_std"])
        ax2.plot(depth, sm, marker="o", color=c, label=f"{short} ({n} layers)")
        ax2.fill_between(depth, sm - ss, sm + ss, color=c, alpha=0.15)

    ax1.axhline(1.0, ls="--", c="grey", lw=1)
    ax1.text(0.02, 1.01, "uniform attention", fontsize=8, color="grey")
    ax1.set_ylim(0, 1.15)
    ax1.set_xlabel("relative depth (0 = first layer, 1 = last)")
    ax1.set_ylabel("normalised row entropy (0 = one token, 1 = uniform)")
    ax1.set_title(f"Attention concentration by depth\n{len(sentences)} sentences, shaded band = 1 sd")
    ax1.legend()
    ax1.grid(alpha=0.3)

    base = results[keys[0]]["baseline"]
    ax2.axhline(base, ls="--", c="grey", lw=1)
    ax2.text(0.02, base + 0.01, f"chance level ({base:.2f})", fontsize=8, color="grey")
    ax2.set_xlabel("relative depth (0 = first layer, 1 = last)")
    ax2.set_ylabel("mean weight on token 0")
    ax2.set_title("Attention sink strength by depth")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT}/e2_depth_profile.png", dpi=150)
    plt.close()

    #per-head sink stability
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (short, r) in zip(axes, results.items()):
        n_layers, n_heads = r["n_layers"], r["n_heads"]
        grid = np.array([[np.mean(r["head_sink"][l][h]) for h in range(n_heads)]
                         for l in range(n_layers)])
        im = ax.imshow(grid, cmap="magma", vmin=0, vmax=1, aspect="auto")
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.set_title(f"{short}: mean sink fraction per head")
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f"{OUT}/e2_head_sink_map.png", dpi=150)
    plt.close()

    #summary
    print("\n--- normalised entropy per layer (0 = one token, 1 = uniform) ---")
    for short, r in results.items():
        print(f"  {short}: {[round(x, 3) for x in r['ent_mean']]}")

    print("\n--- mean weight on token 0 per layer ---")
    for short, r in results.items():
        print(f"  {short}: {[round(x, 3) for x in r['sink_mean']]}")
        print(f"    chance level for this sentence length: {r['baseline']:.3f}")

    print("\n--- per-head sink stability (last layer) ---")
    for short, r in results.items():
        last = r["head_sink"][-1]
        print(f"  {short}:")
        for h, vals in enumerate(last):
            v = np.array(vals)
            print(f"    H{h}: mean {v.mean():.3f}  sd {v.std():.3f}  "
                  f"min {v.min():.3f}  max {v.max():.3f}")

    print(f"\nmean sentence length: "
          f"{results[keys[0]]['mean_len']:.1f} tokens")
    print(f"saved plots to {OUT}/")


if __name__ == "__main__":
    main()
