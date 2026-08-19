"""E1: compare attention patterns across two Pythia sizes on one sentence."""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import matplotlib.pyplot as plt

MODELS = ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m"]
TEXT = "The animal didn't cross the street because it was too tired"


def row_entropy(m):
    """Entropy of each row of an attention matrix."""
    p = m.clamp_min(1e-12)
    return -(p * p.log()).sum(dim=-1)


def load(name):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager")
    model.eval()
    return tok, model


def run(name):
    """Load a model, run TEXT, return tokens and attention."""
    tok, model = load(name)
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    print(f"\n{name}")
    print(f"  layers: {n_layers}  heads: {n_heads}  grids: {n_layers * n_heads}")

    inputs = tok(TEXT, return_tensors="pt")
    tokens = tok.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        out = model(**inputs, output_attentions=True)

    attn = out.attentions  # tuple of n_layers, each [batch, heads, seq, seq]
    print(f"  tokens: {len(tokens)}  shape per layer: {tuple(attn[0].shape)}")

    row_sums = attn[0][0, 0].sum(dim=-1)
    ok = torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    print(f"  rows sum to 1: {ok}")

    return tokens, attn, n_layers, n_heads


def mean_entropy_per_layer(attn):
    """Mean row entropy for each layer, averaged over heads and rows."""
    out = []
    for layer in attn:
        m = layer[0]
        e = torch.stack([row_entropy(m[h]).mean() for h in range(m.shape[0])])
        out.append(e.mean().item())
    return out


def sink_fraction(attn):
    """Mean weight placed on token 0, per layer, averaged over heads and rows."""
    out = []
    for layer in attn:
        m = layer[0]
        out.append(m[:, 1:, 0].mean().item())
    return out


def plot_heads(attn, layer_idx, label, n_heads, path):
    cols = max(1, n_heads // 2)
    rows = 2 if n_heads > 1 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows + 1))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for h in range(n_heads):
        ax = axes[h]
        ax.imshow(attn[layer_idx][0, h].numpy(), cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"{label} H{h}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for h in range(n_heads, len(axes)):
        axes[h].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


results = {}

for name in MODELS:
    tokens, attn, n_layers, n_heads = run(name)
    short = name.split("/")[-1]
    results[short] = {
        "tokens": tokens,
        "entropy": mean_entropy_per_layer(attn),
        "sink": sink_fraction(attn),
        "n_layers": n_layers,
    }
    plot_heads(attn, 0, "L0", n_heads, f"results/{short}_layer0.png")
    plot_heads(attn, -1, "Llast", n_heads, f"results/{short}_lastlayer.png")

names = list(results.keys())
same = results[names[0]]["tokens"] == results[names[1]]["tokens"]
print("\nidentical tokenisation:", same)
if not same:
    print("  ", results[names[0]]["tokens"])
    print("  ", results[names[1]]["tokens"])

#entropy and sink vs relative depth
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

for short, r in results.items():
    n = r["n_layers"]
    depth = [i / (n - 1) for i in range(n)]
    ax1.plot(depth, r["entropy"], marker="o", label=f"{short} ({n} layers)")
    ax2.plot(depth, r["sink"], marker="o", label=f"{short} ({n} layers)")

ax1.set_xlabel("relative depth (0 = first layer, 1 = last)")
ax1.set_ylabel("mean row entropy (nats)")
ax1.set_title("Attention concentration by depth")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.set_xlabel("relative depth (0 = first layer, 1 = last)")
ax2.set_ylabel("mean weight on token 0")
ax2.set_title("Attention sink strength by depth")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/entropy_and_sink_by_depth.png", dpi=150)
plt.close()

print("\nmean row entropy per layer")
for short, r in results.items():
    print(f"  {short}: {[round(x, 2) for x in r['entropy']]}")

print("\nmean weight on token 0 per layer")
for short, r in results.items():
    print(f"  {short}: {[round(x, 3) for x in r['sink']]}")

print("\nsaved plots to results/")