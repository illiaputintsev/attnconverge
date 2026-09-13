"""E7: Size mismatch across model families

Tests whether size mismatch is associated with a larger early-to-late
decline in attention agreement. GPT-2, GPT-2-medium, GPT-Neo-125M and
OPT-125M give three pairs of similar size and three pairs 2.8-2.9x apart.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools
import random
import os
import pickle

MODELS = [
    "gpt2",                      # OpenAI, WebText, 124M
    "gpt2-medium",               # OpenAI, WebText, 355M
    "EleutherAI/gpt-neo-125m",   # EleutherAI, The Pile, 125M
    "facebook/opt-125m",         # Meta, mixed corpus, 125M
]

SIZES = {"gpt2": 124, "gpt2-medium": 355,
         "gpt-neo-125m": 125, "opt-125m": 125}

ORG = {"gpt2": "OpenAI", "gpt2-medium": "OpenAI",
       "gpt-neo-125m": "EleutherAI", "opt-125m": "Meta"}

SENTENCES_PATH = "data/sentences.txt"
OUT = "results"
CACHE = "results/attn_cache_families.pkl"

SINK_CUTOFF = 0.9
N_RANDOM = 150
SEED = 0

COLOURS = ["#c4453a", "#1d9e75", "#d8722c", "#7f77dd", "#2b6cb0", "#8a8578"]


def load_sentences(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def extract(name, sentences):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager")
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads

    probe = tok("The animal was tired", return_tensors="pt")
    probe_toks = tok.convert_ids_to_tokens(probe["input_ids"][0])
    has_bos = probe_toks[0] in (tok.bos_token, "<s>", "</s>")
    print(f"  {name}: {n_layers} layers, {n_heads} heads"
        + ("  (drops a leading BOS token)" if has_bos else ""))

    out, first_tokens = [], None
    for i, text in enumerate(sentences):
        inputs = tok(text, return_tensors="pt")
        toks = tok.convert_ids_to_tokens(inputs["input_ids"][0])
        if has_bos:
            toks = toks[1:]
        if len(toks) < 4:
            continue
        if first_tokens is None:
            first_tokens = toks
        with torch.no_grad():
            res = model(**inputs, output_attentions=True)
        mats = []
        for layer in res.attentions:
            m = layer[0].numpy().astype(np.float16)
            if has_bos:
                m = m[:, 1:, 1:]
                s = m.sum(axis=-1, keepdims=True)
                m = np.divide(m, s, out=np.zeros_like(m), where=s > 0)
            mats.append(m)
        out.append(mats)
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(sentences)}")
    return out, n_layers, n_heads, first_tokens


def get_data(sentences):
    if os.path.exists(CACHE):
        print(f"loading cached attention from {CACHE}")
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    print("extracting attention (first run)")
    data = {name: extract(name, sentences) for name in MODELS}
    os.makedirs(OUT, exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(data, f)
    print(f"cached to {CACHE}")
    return data


def lower_triangle(m):
    """Causal entries, excluding row and column 0."""
    seq = m.shape[0]
    vals = []
    for i in range(1, seq):
        for j in range(1, i + 1):
            vals.append(m[i, j])
    return np.array(vals, dtype=np.float32)


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def sink_fraction(attn, n_layers, n_heads):
    """Mean weight on token 0 per head, excluding the first query."""
    acc = np.zeros((n_layers, n_heads))
    for sent in attn:
        for l in range(n_layers):
            acc[l] += sent[l][:, 1:, 0].astype(np.float32).mean(axis=1)
    return acc / len(attn)


def head_similarity(attnA, attnB, lA, lB, hA, hB):
    """Mean cosine over paired sentences of equal token length"""
    scores = []
    for sA, sB in zip(attnA, attnB):
        a = lower_triangle(sA[lA][hA])
        b = lower_triangle(sB[lB][hB])
        if len(a) != len(b) or len(a) == 0:
            continue
        scores.append(cosine(a, b))
    return float(np.mean(scores)) if scores else 0.0


def best_match(attnA, attnB, lA, lB, liveA, liveB):
    """Mean best match in B for each eligible head in A"""
    if not liveA or not liveB:
        return np.nan
    return float(np.mean([
        max(head_similarity(attnA, attnB, lA, lB, hA, hB) for hB in liveB)
        for hA in liveA
    ]))


def random_baseline(attnA, attnB, LA, HA, LB, HB, sinkA, sinkB, cutoff):
    """Mean similarity of eligible random head pairs across both stacks."""
    scores, tries = [], 0
    while len(scores) < N_RANDOM and tries < N_RANDOM * 12:
        tries += 1
        lA, hA = random.randrange(LA), random.randrange(HA)
        lB, hB = random.randrange(LB), random.randrange(HB)
        if sinkA[lA, hA] > cutoff or sinkB[lB, hB] > cutoff:
            continue
        scores.append(head_similarity(attnA, attnB, lA, lB, hA, hB))
    return float(np.mean(scores)) if scores else np.nan


def compare(nameA, nameB, data, sinks):
    """One pair: gap over baseline at each relative depth."""
    attnA, LA, HA, _ = data[nameA]
    attnB, LB, HB, _ = data[nameB]
    sinkA, sinkB = sinks[nameA], sinks[nameB]

    shortA, shortB = nameA.split("/")[-1], nameB.split("/")[-1]
    ratio = max(SIZES[shortA], SIZES[shortB]) / min(SIZES[shortA], SIZES[shortB])
    same_org = ORG[shortA] == ORG[shortB]
    kind = "same org" if same_org else "different orgs"
    print(f"\n--- {shortA} vs {shortB}  ({ratio:.1f}x apart, {kind}) ---")

    depthsA = [i / (LA - 1) for i in range(LA)]
    depthsB = [i / (LB - 1) for i in range(LB)]

    rows = []
    for lA, dA in enumerate(depthsA):
        lB = int(np.argmin([abs(dA - dB) for dB in depthsB]))
        liveA = [h for h in range(HA) if sinkA[lA, h] <= SINK_CUTOFF]
        liveB = [h for h in range(HB) if sinkB[lB, h] <= SINK_CUTOFF]
        best = best_match(attnA, attnB, lA, lB, liveA, liveB)
        rand = random_baseline(attnA, attnB, LA, HA, LB, HB,
                               sinkA, sinkB, SINK_CUTOFF)
        rows.append({"depth": dA, "best": best, "rand": rand,
                     "nA": len(liveA), "nB": len(liveB)})
        print(f"  depth {dA:.2f}  L{lA}/L{lB}  heads {len(liveA)}x{len(liveB)}  "
              f"best {best:.3f}  random {rand:.3f}  gap {best - rand:+.3f}")

    gaps = [r["best"] - r["rand"] for r in rows if not np.isnan(r["best"])]
    early = float(np.mean(gaps[:2]))
    late = float(np.mean(gaps[-3:]))          # last three layers
    print(f"  early gap {early:+.3f}   late gap {late:+.3f}   "
          f"decay {early - late:+.3f}")

    return {"rows": rows, "shortA": shortA, "shortB": shortB, "ratio": ratio,
            "same_org": same_org, "early": early, "late": late,
            "decay": early - late, "mean": float(np.mean(gaps))}


def main():
    os.makedirs(OUT, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)

    sentences = load_sentences(SENTENCES_PATH)
    print(f"{len(sentences)} sentences, {len(MODELS)} models")

    data = get_data(sentences)

    toks = [data[m][3] for m in MODELS]
    same = all(t == toks[0] for t in toks)
    print(f"\nfirst-sentence tokenisation matches after BOS handling: {same}")
    if not same:
        for m in MODELS:
            print(f"  {m.split('/')[-1]}: {data[m][3]}")
        print("  matrices are not comparable; stopping")
        return

    sinks = {m: sink_fraction(data[m][0], data[m][1], data[m][2]) for m in MODELS}
    print(f"\nheads under the {SINK_CUTOFF} sink cutoff:")
    for m in MODELS:
        s = sinks[m]
        print(f"  {m.split('/')[-1]}: {(s <= SINK_CUTOFF).sum()} of {s.size}")

    print("\n=== pairwise comparisons ===")
    results = {}
    for a, b in itertools.combinations(MODELS, 2):
        results[(a, b)] = compare(a, b, data, sinks)

    print("\n=== early-to-late gap by size ratio ===")
    equal = [r for r in results.values() if r["ratio"] < 1.2]
    unequal = [r for r in results.values() if r["ratio"] >= 1.2]

    print(f"\n  {'pair':<32} {'ratio':>7}  {'early':>7}  {'late':>7}  {'decay':>7}")
    for r in sorted(results.values(), key=lambda x: x["ratio"]):
        pair = f"{r['shortA']} vs {r['shortB']}"
        print(f"  {pair:<32} {r['ratio']:6.1f}x  {r['early']:+7.3f}  "
              f"{r['late']:+7.3f}  {r['decay']:+7.3f}")

    if equal and unequal:
        eq_decay = float(np.mean([r["decay"] for r in equal]))
        un_decay = float(np.mean([r["decay"] for r in unequal]))
        eq_late = float(np.mean([r["late"] for r in equal]))
        un_late = float(np.mean([r["late"] for r in unequal]))
        print(f"\n  near-equal-size pairs: mean decay {eq_decay:+.3f}, "
              f"late gap {eq_late:+.3f}  ({len(equal)} pairs)")
        print(f"  unequal-size pairs:    mean decay {un_decay:+.3f}, "
              f"late gap {un_late:+.3f}  ({len(unequal)} pairs)")
        print(f"\n  unequal minus near-equal: mean decay {un_decay - eq_decay:+.3f}, "
              f"late gap {un_late - eq_late:+.3f}")

    print("\n=== early gap across organisations ===")
    cross = [r for r in results.values() if not r["same_org"]]
    within = [r for r in results.values() if r["same_org"]]
    for r in cross:
        print(f"  {r['shortA']} vs {r['shortB']}: early {r['early']:+.3f}")
    if within:
        for r in within:
            print(f"  {r['shortA']} vs {r['shortB']} (same org): "
                  f"early {r['early']:+.3f}")
    print(f"\n  mean early gap across organisations: "
          f"{np.mean([r['early'] for r in cross]):+.3f}  ({len(cross)} pairs)")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

    ax = axes[0]
    for k, r in enumerate(results.values()):
        d = [x["depth"] for x in r["rows"]]
        g = [x["best"] - x["rand"] for x in r["rows"]]
        style = "-" if r["ratio"] < 1.2 else "--"
        ax.plot(d, g, marker="o", ms=4, ls=style, color=COLOURS[k % len(COLOURS)],
                label=f"{r['shortA']} vs {r['shortB']} ({r['ratio']:.1f}x)")
    ax.axhline(0, color="#999", lw=1, ls=":")
    ax.set_xlabel("relative depth")
    ax.set_ylabel("gap over random baseline")
    ax.set_title("Solid = near-equal sizes, dashed = unequal")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    xs = [r["ratio"] for r in results.values()]
    ys = [r["decay"] for r in results.values()]
    cs = ["#1d9e75" if r["same_org"] else "#c4453a" for r in results.values()]
    ax.scatter(xs, ys, s=80, c=cs, zorder=3)
    for r in results.values():
        ax.annotate(f"{r['shortA'].replace('gpt-','').replace('-125m','')}/"
                    f"{r['shortB'].replace('gpt-','').replace('-125m','')}",
                    (r["ratio"], r["decay"]), fontsize=7.5,
                    xytext=(5, 4), textcoords="offset points")
    ax.axhline(0, color="#999", lw=1, ls=":")
    ax.set_xlabel("size ratio between the two models")
    ax.set_ylabel("decay from early to late layers")
    ax.set_title("Green = same organisation, red = different")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT}/e7_families.png", dpi=150)
    plt.close()

    print(f"\nsaved plot to {OUT}/e7_families.png")


if __name__ == "__main__":
    main()
