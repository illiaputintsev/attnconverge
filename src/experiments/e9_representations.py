"""E9: Attention and representation agreement

Compares attention profiles with linear CKA of final-token hidden states at
corresponding relative depths. The exact permutation expectation is subtracted
from CKA to account for its finite-sample baseline. Ordinary and repeated
passages are evaluated separately.

Final-token states represent the same complete text despite differences in
tokenisation. Attention comparisons use the subset with identical token
boundaries, with head matches fixed on a separate selection set. CKA is then
recomputed on that subset so both profiles use the same passages.
"""

import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np

from src.attnlib.matching import evaluate_matches, fit_matches, stable_seed
from src.attnlib.representations import compare_representations
from src.attnlib.extract import MODEL_INFO, AttentionCache, load_passages, split_passages


MODELS = [
    "pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b",
    "gpt2", "gpt2-medium", "gpt-neo-125m", "opt-125m",
    "crfm-gpt2-x21", "crfm-gpt2-x49",
]
SELECTION = "data/selection.txt"
EVALUATION = "data/evaluation.txt"
REPETITION = "data/repetition.txt"
N_TOKENS = 32
CACHE = "results/attention_cache"
OUT = "results/e9"
MIN_EXAMPLES = 25
SEED = 20260911
SINK_CUTOFF = 0.9
N_PERMUTATIONS = 200
BANDS = ("early", "late")


def compare_attention(name_a, name_b, store):
    """Fit heads on selection passages and score the aligned evaluation subset."""
    accepted, _, _ = store.aligned_ids([name_a, name_b])
    selection, conditions = split_passages(store, accepted)
    ordinary = conditions["ordinary"]
    result = {"selection_ids": selection, "ids": ordinary, "scores": None}
    if len(selection) < MIN_EXAMPLES or len(ordinary) < MIN_EXAMPLES:
        return result

    a, b = store.load(name_a, selection), store.load(name_b, selection)
    seed = stable_seed(SEED, name_a, name_b)
    plan = fit_matches(a["attention"], b["attention"],
                       a["sink_start"].mean(axis=0), b["sink_start"].mean(axis=0),
                       cutoff=SINK_CUTOFF, seed=seed)
    del a, b

    a, b = store.load(name_a, ordinary), store.load(name_b, ordinary)
    # The different-input mean is exact; no sampled shuffles are needed here.
    result["scores"] = evaluate_matches(plan, a["attention"], b["attention"],
                                        seed=seed + 1, n_shuffles=0, n_bootstrap=0)
    return result


def compare(name_a, name_b, store, ordinary, repetition):
    """Compare representations on all passages and on the attention subset."""
    seed = stable_seed(SEED, "representations", name_a, name_b)
    hidden_a, hidden_b = (store.load_hidden(name, ordinary) for name in (name_a, name_b))
    full = compare_representations(hidden_a, hidden_b, seed=seed,
                                   n_permutations=N_PERMUTATIONS)
    repeated = None
    if len(repetition) >= MIN_EXAMPLES:
        a, b = (store.load_hidden(name, repetition) for name in (name_a, name_b))
        repeated = compare_representations(a, b, seed=seed + 1,
                                            n_permutations=N_PERMUTATIONS)

    attention = compare_attention(name_a, name_b, store)
    aligned = None
    if attention["scores"] is not None:
        index = {sample_id: i for i, sample_id in enumerate(ordinary)}
        rows = [index[i] for i in attention["ids"]]
        aligned = compare_representations(hidden_a[rows], hidden_b[rows], seed=seed + 2,
                                           n_permutations=N_PERMUTATIONS)

    pa, pb = (store.metadata[n]["parameter_count"] / 1e6 for n in (name_a, name_b))
    row = {"model_a": name_a, "model_b": name_b, "n_full": len(ordinary),
           "n_aligned": aligned["n_examples"] if aligned else 0,
           "n_repetition": repeated["n_examples"] if repeated else 0,
           "same_family": MODEL_INFO[name_a]["family"] == MODEL_INFO[name_b]["family"],
           "same_organisation": MODEL_INFO[name_a]["organisation"] == MODEL_INFO[name_b]["organisation"],
           "smaller_size_m": min(pa, pb), "size_ratio": max(pa, pb) / min(pa, pb)}
    for band in BANDS:
        summary = full["summaries"][band]
        row[f"{band}_cka"] = summary["raw_cka"]
        row[f"{band}_cka_null"] = summary["shuffled_null_mean"]
        row[f"{band}_cka_excess"] = summary["excess_cka"]
        row[f"{band}_cka_p"] = summary["permutation_pvalue"]
        row[f"{band}_cka_excess_aligned"] = aligned["summaries"][band]["excess_cka"] if aligned else None
        row[f"{band}_cka_excess_repetition"] = repeated["summaries"][band]["excess_cka"] if repeated else None
        scores = attention["scores"]["symmetric_bands"][band] if attention["scores"] else None
        row[f"{band}_attention_gap"] = scores["matched_minus_random"] if scores else None
        row[f"{band}_attention_input_excess"] = scores["same_minus_shuffled"] if scores else None
    return {"summary": row, "full": full, "aligned": aligned,
            "repetition": repeated, "attention": attention}


def _depth_profile_attention(result):
    rows = result["directions"][0]["rows"]
    return [(r["depth"], r["matched_minus_random"], r["same_minus_shuffled"]) for r in rows
            if r["matched_minus_random"] is not None]


def _depth_profile_representation(result):
    rows = [r for r in result["rows"] if r["direction"] == "a_to_b" and r["excess_cka"] is not None]
    return [(r["depth"], r["excess_cka"], r["raw_cka"]) for r in rows]


def _number(value):
    return f"{value:+.3f}" if value is not None else "n/a"


def print_comparison(result):
    row = result["summary"]
    attention = result["attention"]
    print(f"\n{row['model_a']} vs {row['model_b']}", flush=True)
    print(f"  aligned passages: {len(attention['selection_ids'])} selection, "
          f"{len(attention['ids'])} evaluation")
    for label, key in (("ordinary", "full"), ("repetition", "repetition"),
                       ("attention subset", "aligned")):
        scores = result[key]
        if scores is None:
            print(f"  {label}: skipped (minimum {MIN_EXAMPLES} passages per required split)")
            continue
        print(f"  {label}: n={scores['n_examples']}")
        for band in BANDS:
            s = scores["summaries"][band]
            print(f"    {band:5s} CKA {_number(s['raw_cka'])}, "
                  f"permutation mean {_number(s['shuffled_null_mean'])}, "
                  f"excess {_number(s['excess_cka'])}")
    if attention["scores"] is not None:
        for band in BANDS:
            print(f"  {band:5s} attention gap {_number(row[band + '_attention_gap'])}, "
                  f"same minus different input {_number(row[band + '_attention_input_excess'])}, "
                  f"aligned excess CKA {_number(row[band + '_cka_excess_aligned'])}")


def summarise(rows):
    """Descriptive pair correlations and within- versus cross-family means."""
    def corr(xk, yk):
        pts = [(r[xk], r[yk]) for r in rows if r.get(xk) is not None and r.get(yk) is not None]
        if len(pts) < 3:
            return {"n_pairs": len(pts), "pearson_r": None}
        x, y = np.asarray(pts, dtype=float).T
        r = float(np.corrcoef(x, y)[0, 1]) if np.ptp(x) > 0 and np.ptp(y) > 0 else None
        return {"n_pairs": len(pts), "pearson_r": r}
    def mean(key, predicate):
        v = [r[key] for r in rows if r.get(key) is not None and predicate(r)]
        return float(np.mean(v)) if v else None
    return {"late_attention_gap_vs_late_cka_excess": corr("late_attention_gap", "late_cka_excess_aligned"),
            "late_attention_input_excess_vs_late_cka_excess": corr("late_attention_input_excess", "late_cka_excess_aligned"),
            "mean_late_cka_excess": {"same_family": mean("late_cka_excess", lambda r: r["same_family"]),
                                     "cross_family": mean("late_cka_excess", lambda r: not r["same_family"])},
            "mean_early_cka_excess": {"same_family": mean("early_cka_excess", lambda r: r["same_family"]),
                                      "cross_family": mean("early_cka_excess", lambda r: not r["same_family"])}}


def _style():
    """Settings scoped to these figures, leaving other experiments unchanged."""
    return {
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "font.family": "DejaVu Sans",
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 12,
        "axes.titleweight": "semibold", "axes.titlepad": 12,
        "text.color": "#243342", "axes.labelcolor": "#243342",
        "xtick.color": "#536170", "ytick.color": "#536170",
        "axes.edgecolor": "#d6dde3", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": False,
        "grid.color": "#e7ecf0", "grid.linewidth": .7,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    }


def _model_label(name):
    return (name.replace("pythia-", "Pythia ").replace("gpt2-medium", "GPT-2 medium")
            .replace("gpt2", "GPT-2").replace("gpt-neo-", "GPT-Neo ")
            .replace("opt-", "OPT ").replace("70m", "70M").replace("160m", "160M")
            .replace("410m", "410M").replace("125m", "125M").replace("1b", "1B"))


def _save_figure(fig, path):
    path = Path(path)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    if path.suffix.lower() != ".pdf":
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def plot_summary(path, rows, names, title):
    settings = _style()
    index = {n: i for i, n in enumerate(names)}
    mats = {band: np.full((len(names), len(names)), np.nan) for band in BANDS}
    for r in rows:
        i, j = sorted((index[r["model_a"]], index[r["model_b"]]), reverse=True)
        for band in BANDS:
            value = r.get(f"{band}_cka_excess")
            if value is not None:
                mats[band][i, j] = value
    finite = np.concatenate([m[np.isfinite(m)] for m in mats.values()])
    limit = max(.05, float(np.max(np.abs(finite))) if len(finite) else .05)
    limit = float(np.ceil(limit * 20) / 20)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    with plt.rc_context(settings):
        fig, axes = plt.subplots(1, 2, figsize=(max(12, len(names) * 1.75), 7.5))
        fig.subplots_adjust(left=.11, right=.9, top=.82, bottom=.23, wspace=.42)
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("white")
        labels = [_model_label(n) for n in names]
        for ax, band, letter in zip(axes, BANDS, ("a", "b")):
            m = mats[band]
            im = ax.imshow(np.ma.masked_invalid(m), cmap=cmap, norm=norm)
            ax.set_xticks(range(len(names)), labels, rotation=45, ha="right", rotation_mode="anchor")
            ax.set_yticks(range(len(names)), labels)
            ax.tick_params(length=0, labelsize=9, pad=7)
            for spine in ax.spines.values():
                spine.set_visible(False)
            depth = "0–25% of relative depth" if band == "early" else "75–100% of relative depth"
            ax.set_title(f"{letter}  {band.capitalize()} layers", loc="left", pad=28)
            ax.text(0, 1.035, depth, transform=ax.transAxes, fontsize=9, color="#6b7785")
            for i in range(len(names)):
                for j in range(i):
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           edgecolor="white", linewidth=1.7))
                    if np.isfinite(m[i, j]):
                        rgba = cmap(norm(m[i, j]))
                        luminance = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
                        ax.text(j, i, f"{m[i, j]:+.3f}", ha="center", va="center", fontsize=8.5,
                                color="white" if luminance < .5 else "#243342")
                    else:
                        ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor="#eef1f4",
                                               edgecolor="white", linewidth=1.7))
                        ax.text(j, i, "—", ha="center", va="center", color="#8a96a2")
        colour_ax = fig.add_axes([.93, .28, .014, .45])
        fig.colorbar(im, cax=colour_ax, label="Excess CKA · shared scale", extend="neither")
        fig.suptitle(title, x=.055, y=.98, ha="left", fontsize=17, fontweight="semibold")
        fig.text(.055, .92, "Final-token representations · observed linear CKA minus its exact permutation expectation",
                 fontsize=10.5, color="#536170")
        fig.text(.055, .055, "One cell per unordered model pair; both depth-mapping directions are averaged. Diagonal omitted.",
                 fontsize=9, color="#536170")
        fig.text(.055, .028, "Negative values are retained. A grey cell indicates an unavailable comparison.",
                 fontsize=9, color="#536170")
        _save_figure(fig, path)
        plt.close(fig)


def _scatter_pair_labels(ax, points):
    """Place compact IDs around unchanged points, avoiding label collisions."""
    fig = ax.figure
    fig.canvas.draw()
    display_points = ax.transData.transform(points)
    occupied = []
    bounds = ax.get_window_extent()
    for number, (x, y) in enumerate(display_points, 1):
        width, height = (15 if number < 10 else 22), 17
        candidates = []
        for distance in (13, 23, 35, 50, 70, 95):
            for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                cx, cy = x + distance * np.cos(angle), y + distance * np.sin(angle)
                rectangle = (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)
                l, b, r, t = rectangle
                outside = max(bounds.x0 - l, 0) + max(r - bounds.x1, 0) + max(bounds.y0 - b, 0) + max(t - bounds.y1, 0)
                overlaps = sum(max(0, min(r, q[2]) - max(l, q[0])) * max(0, min(t, q[3]) - max(b, q[1]))
                               for q in occupied)
                hits = sum(l - 5 < px < r + 5 and b - 5 < py < t + 5 for px, py in display_points)
                candidates.append((outside * 1e5 + overlaps * 1e3 + hits * 1e4 + distance, cx, cy, rectangle))
        _, cx, cy, rectangle = min(candidates, key=lambda candidate: candidate[0])
        occupied.append(rectangle)
        ax.annotate(str(number), xy=points[number - 1],
                    xytext=((cx - x) * 72 / fig.dpi, (cy - y) * 72 / fig.dpi), textcoords="offset points",
                    ha="center", va="center", fontsize=8, color="#344354", zorder=5,
                    bbox={"boxstyle": "round,pad=.13", "fc": "white", "ec": "none", "alpha": .93},
                    arrowprops={"arrowstyle": "-", "color": "#a1acb7", "lw": .55, "shrinkA": 3, "shrinkB": 4})


def plot_attention_against_representation(path, rows):
    settings = _style()
    pts = [r for r in rows if r.get("late_attention_gap") is not None and r.get("late_cka_excess_aligned") is not None]
    with plt.rc_context(settings):
        fig = plt.figure(figsize=(14, max(7.8, 2.0 + .22 * len(pts))))
        ax = fig.add_axes([.075, .18, .49, .65])
        key_ax = fig.add_axes([.63, .13, .35, .72])
        key_ax.axis("off")
        if not pts:
            ax.text(.5, .5, "No pair has enough aligned passages\nfor both comparisons",
                    ha="center", va="center", transform=ax.transAxes)
        else:
            for same, label, colour, marker in ((True, "Same family", "#007c83", "o"),
                                                  (False, "Different families", "#c26735", "D")):
                group = [r for r in pts if r["same_family"] == same]
                if group:
                    ax.scatter([r["late_attention_gap"] for r in group], [r["late_cka_excess_aligned"] for r in group],
                               s=46, color=colour, marker=marker, edgecolor="white", linewidth=.7, label=label, zorder=3)
            ax.margins(x=.15, y=.17)
            ax.legend(loc="upper left", bbox_to_anchor=(0, 1.10), ncol=2, fontsize=9, handletextpad=.4)
            key_ax.text(0, 1, f"PAIR KEY  ·  {len(pts)} comparisons", va="top", fontsize=10, fontweight="semibold")
            step = min(.038, .92 / max(len(pts), 1))
            for number, row in enumerate(pts, 1):
                colour = "#007c83" if row["same_family"] else "#c26735"
                y = .94 - (number - 1) * step
                key_ax.text(0, y, f"{number:02d}", va="center", color=colour, fontsize=8.5, fontweight="semibold")
                key_ax.text(.075, y, f"{_model_label(row['model_a'])}  /  {_model_label(row['model_b'])}",
                            va="center", fontsize=8.5)
        ax.axvline(0, color="#c3cbd3", lw=.8, ls="--", zorder=0)
        ax.axhline(0, color="#c3cbd3", lw=.8, ls="--", zorder=0)
        ax.grid(axis="both", color="#e7ecf0", linewidth=.7, zorder=0)
        ax.set_xlabel("Attention agreement\nHeld-out matched minus random cosine", labelpad=10)
        ax.set_ylabel("Representation agreement\nCKA minus permutation expectation", labelpad=10)
        fig.suptitle("Late-layer attention and representation agreement", x=.075, y=.97,
                     ha="left", fontsize=17, fontweight="semibold")
        fig.text(.075, .91, "Each point compares the same aligned held-out passages within one model pair.",
                 fontsize=10.5, color="#536170")
        fig.text(.075, .065, "Point IDs refer to the pair key. Coordinates are unchanged; labels are displaced for readability.",
                 fontsize=9, color="#536170")
        fig.text(.075, .035, "Pairs share model checkpoints and may use different passage subsets. The two metrics have different scales.",
                 fontsize=9, color="#536170")
        if pts:
            _scatter_pair_labels(ax, [(r["late_attention_gap"], r["late_cka_excess_aligned"]) for r in pts])
        _save_figure(fig, path)
        plt.close(fig)


def plot_profile(path, name_a, name_b, attention_rows, representation_rows, aligned):
    settings = _style()
    with plt.rc_context(settings):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        ax = axes[0]
        if attention_rows:
            d, gap, excess = zip(*attention_rows)
            ax.plot(d, gap, marker="o", color="#007c83", label="Matched minus random")
            ax.plot(d, excess, marker="o", color="#c26735", label="Same minus different input")
        ax.axhline(0, color="#a1acb7", lw=.8, ls="--")
        ax.set_xlabel("Relative depth")
        ax.set_ylabel("Cosine difference")
        ax.set_title(f"Attention · {_model_label(name_a)} / {_model_label(name_b)}", loc="left")
        ax.legend(fontsize=9)
        ax = axes[1]
        if representation_rows:
            d, excess, raw = zip(*representation_rows)
            ax.plot(d, raw, marker="o", color="#8b99a6", label="Raw CKA")
            ax.plot(d, excess, marker="o", color="#6955a5", label="CKA minus permutation expectation")
        ax.axhline(0, color="#a1acb7", lw=.8, ls="--")
        ax.set_xlabel("Relative depth")
        ax.set_ylabel("CKA")
        ax.set_title("Representation · " + ("same passages" if aligned else "all held-out passages"), loc="left")
        ax.legend(fontsize=9)
        for ax in axes:
            ax.grid(axis="y")
        fig.tight_layout()
        _save_figure(fig, path)
        plt.close(fig)


def main():
    if len(set(MODELS)) < 2 or len(MODELS) != len(set(MODELS)):
        raise ValueError("MODELS must contain at least two distinct models")
    if MIN_EXAMPLES < 4:
        raise ValueError("CKA requires at least four passages")
    store = AttentionCache(load_passages(SELECTION, EVALUATION, REPETITION, N_TOKENS), CACHE, MODELS)
    _, cohorts = split_passages(store, list(store.records))
    ordinary, repetition = cohorts["ordinary"], cohorts["repetition"]
    if len(ordinary) < MIN_EXAMPLES:
        raise ValueError("Too few evaluation passages for representation comparison")
    output = Path(OUT).expanduser()
    (output / "profiles").mkdir(parents=True, exist_ok=True)

    results = []
    for name_a, name_b in itertools.combinations(MODELS, 2):
        result = compare(name_a, name_b, store, ordinary, repetition)
        print_comparison(result)
        if result["aligned"] is not None:
            plot_profile(output / "profiles" / f"{name_a}__{name_b}.png", name_a, name_b,
                         _depth_profile_attention(result["attention"]["scores"]),
                         _depth_profile_representation(result["aligned"]), True)
        results.append(result)

    rows = [r["summary"] for r in results]
    plot_summary(output / "summary.png", rows, MODELS,
                 f"E9: representation agreement, {len(ordinary)} held-out passages")
    plot_attention_against_representation(output / "attention_vs_representation.png", rows)
    summary = summarise(rows)
    print("\n=== descriptive pair summaries ===")
    for band in BANDS:
        means = summary[f"mean_{band}_cka_excess"]
        print(f"  {band:5s} mean excess CKA: same family {_number(means['same_family'])}, "
              f"different families {_number(means['cross_family'])}")
    for label, key in (("attention gap", "late_attention_gap_vs_late_cka_excess"),
                       ("input excess", "late_attention_input_excess_vs_late_cka_excess")):
        correlation = summary[key]
        print(f"  late {label} vs aligned excess CKA: "
              f"r={_number(correlation['pearson_r'])}, n={correlation['n_pairs']} pairs")
    print(f"\n{len(rows)} pairs compared; figures saved to {output}", flush=True)
    return results


if __name__ == "__main__":
    main()
