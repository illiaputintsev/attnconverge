"""E8: Input dependence of attention agreement

Head matches are selected by mean cosine similarity at corresponding relative
depths, then fixed for evaluation on held-out passages. Each match is scored
on the same passage and on all different-passage pairings, with fixed random
head pairs as a baseline. Direct comparisons require identical token boundaries.

Results are compared across model sizes, size ratios and sink prevalence.
A paired repetition condition tests how agreement changes with repeated text.
"""

import itertools
import numpy as np
from pathlib import Path

from src.attnlib.matching import evaluate_matches, fit_matches, stable_seed
from src.attnlib.extract import MODEL_INFO, AttentionCache, load_passages, split_passages
from src.attnlib.metrics import paired_structure_difference, sink_summary


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
OUT = "results/e8"

MIN_EXAMPLES = 25
SEED = 20260911
SINK_CUTOFF = 0.9
SENSITIVITY_CUTOFFS = (0.7, 1.0)
N_BOOTSTRAPS = 300


def score_conditions(name_a, name_b, store, conditions, plans, seed):
    """Evaluate the selected heads on ordinary and repeated passages."""
    scores, sensitivity = {}, []
    for condition, ids in conditions.items():
        if len(ids) < MIN_EXAMPLES:
            scores[condition] = {"status": "deferred_insufficient_aligned_examples", "ids": ids}
            continue

        a, b = store.load(name_a, ids), store.load(name_b, ids)
        result = evaluate_matches(
            plans[str(SINK_CUTOFF)], a["attention"], b["attention"],
            seed=seed + 1, n_shuffles=0, n_bootstrap=N_BOOTSTRAPS,
        )
        scores[condition] = {"status": "complete", "ids": ids, "result": compact_result(result)}

        if condition == "ordinary":
            for cutoff, plan in plans.items():
                secondary = result if float(cutoff) == SINK_CUTOFF else evaluate_matches(
                    plan, a["attention"], b["attention"], seed=seed + 1,
                    n_shuffles=0, n_bootstrap=0,
                )
                bands = secondary["symmetric_bands"]
                sensitivity.append({
                    "model_a": name_a, "model_b": name_b, "cutoff": float(cutoff),
                    "early_gap": bands["early"]["matched_minus_random"],
                    "late_gap": bands["late"]["matched_minus_random"],
                    "late_input_excess": bands["late"]["same_minus_shuffled"],
                    "n_evaluation": len(ids),
                })
        del a, b
    return scores, sensitivity


def compare(name_a, name_b, store, common_ids):
    """Select head matches, evaluate both directions and summarise one model pair."""
    accepted, rejected, _ = store.aligned_ids([name_a, name_b])
    selection_ids, conditions = split_passages(store, accepted)
    coverage = {
        "model_a": name_a, "model_b": name_b,
        "n_selection": len(selection_ids), "n_evaluation": len(conditions["ordinary"]),
        "n_paired_structure": len(conditions["repetition"]), "n_rejected": len(rejected),
        "alignment": "exact", "status": "deferred_insufficient_aligned_examples",
    }
    if len(selection_ids) < MIN_EXAMPLES or len(conditions["ordinary"]) < MIN_EXAMPLES:
        return {"coverage": coverage, "selection_ids": selection_ids,
                "conditions": conditions, "rejected_ids": rejected}

    # Sink exclusions and both head mappings are fixed before evaluation.
    selected_a = store.load(name_a, selection_ids)
    selected_b = store.load(name_b, selection_ids)
    sink_a, sink_b = (x["sink_start"].mean(axis=0) for x in (selected_a, selected_b))
    seed = stable_seed(SEED, name_a, name_b)
    cutoffs = dict.fromkeys([SINK_CUTOFF, *SENSITIVITY_CUTOFFS])
    plans = {
        str(c): fit_matches(selected_a["attention"], selected_b["attention"],
                            sink_a, sink_b, cutoff=c, seed=seed)
        for c in cutoffs
    }
    sinks = {name_a: sink_summary(selected_a, SINK_CUTOFF),
             name_b: sink_summary(selected_b, SINK_CUTOFF)}
    del selected_a, selected_b

    scores, sensitivity = score_conditions(name_a, name_b, store, conditions, plans, seed)
    structure = None
    if all(scores[c]["status"] == "complete" for c in ("ordinary_paired", "repetition")):
        structure = paired_structure_difference(
            scores["ordinary_paired"]["result"], scores["repetition"]["result"],
            seed=seed + 2, n_bootstrap=N_BOOTSTRAPS,
        )

    # Also score the evaluation passages aligned across every model in the set.
    common_score = None
    if len(common_ids) >= MIN_EXAMPLES:
        a, b = store.load(name_a, common_ids), store.load(name_b, common_ids)
        common_score = compact_result(evaluate_matches(
            plans[str(SINK_CUTOFF)], a["attention"], b["attention"],
            seed=seed + 3, n_shuffles=0, n_bootstrap=0,
        ))
        del a, b

    bands = scores["ordinary"]["result"]["symmetric_bands"]
    pa, pb = (store.metadata[n]["parameter_count"] / 1e6 for n in (name_a, name_b))
    summary = {
        "model_a": name_a, "model_b": name_b, "n_selection": len(selection_ids),
        "n_evaluation": len(conditions["ordinary"]),
        "n_paired_structure": len(conditions["repetition"]),
        "early_gap": bands["early"]["matched_minus_random"],
        "late_gap": bands["late"]["matched_minus_random"],
        "late_input_excess": bands["late"]["same_minus_shuffled"],
        "early_input_excess": bands["early"]["same_minus_shuffled"],
        "smaller_size_m": min(pa, pb), "size_ratio": max(pa, pb) / min(pa, pb),
        "sink_share_a": sinks[name_a]["parked_share"],
        "sink_share_b": sinks[name_b]["parked_share"],
        "same_family": MODEL_INFO[name_a]["family"] == MODEL_INFO[name_b]["family"],
        "same_organisation": MODEL_INFO[name_a]["organisation"] == MODEL_INFO[name_b]["organisation"],
    }
    coverage["status"] = "complete"
    return {
        "coverage": coverage, "summary": summary,
        "selection_ids": selection_ids, "rejected_ids": rejected,
        "plans": plans, "selection_sinks": sinks, "conditions": scores,
        "structure_change": structure, "sensitivity_summary": sensitivity,
        "common_cohort_ids": common_ids, "common_cohort_result": common_score,
    }


def _finite(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _mean(rows, key):
    values = [_finite(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def _correlation(rows, predictor):
    paired = []
    for row in rows:
        x, y = predictor(row), _finite(row.get("late_gap"))
        if x is not None and y is not None:
            paired.append((x, y))
    result = {"n_pairs": len(paired), "pearson_r": None}
    if len(paired) >= 2:
        x, y = np.asarray(paired, dtype=float).T
        # Constant inputs have no defined correlation.
        if np.ptp(x) > 0 and np.ptp(y) > 0:
            result["pearson_r"] = float(np.corrcoef(x, y)[0, 1])
    return result


def _log_predictor(row, key):
    value = _finite(row.get(key))
    return float(np.log10(value)) if value is not None and value > 0 else None


def _sink_predictor(row):
    a, b = _finite(row.get("sink_share_a")), _finite(row.get("sink_share_b"))
    return (a + b) / 2 if a is not None and b is not None else None


def summarise(pair_rows, model_names):
    """Per-model late means and pairwise correlations with size, ratio and sinks."""
    rows, names = list(pair_rows), list(model_names)
    if len(set(names)) != len(names):
        raise ValueError("model_names must be unique")
    per_model = []
    for name in names:
        selected = [r for r in rows if name in (r.get("model_a"), r.get("model_b"))]
        partners = {
            r.get("model_b") if r.get("model_a") == name else r.get("model_a")
            for r in selected
        }
        partners.discard(None)
        partners.discard(name)
        per_model.append({
            "model": name, "n_partners": len(partners), "n_pairs": len(selected),
            "n_late_gap_pairs": sum(_finite(r.get("late_gap")) is not None for r in selected),
            "n_late_input_excess_pairs": sum(_finite(r.get("late_input_excess")) is not None for r in selected),
            "mean_late_gap": _mean(selected, "late_gap"),
            "mean_late_input_excess": _mean(selected, "late_input_excess"),
        })
    return {
        "n_pairs": len(rows), "per_model": per_model,
        "correlations": {
            "log10_smaller_size_vs_late_gap": _correlation(rows, lambda r: _log_predictor(r, "smaller_size_m")),
            "log10_size_ratio_vs_late_gap": _correlation(rows, lambda r: _log_predictor(r, "size_ratio")),
            "mean_sink_share_vs_late_gap": _correlation(rows, _sink_predictor),
        },
    }


def compact_result(result):
    # Bands retain paired-example measurements; per-layer examples are redundant.
    for direction in result["directions"]:
        for row in direction["rows"]:
            row.pop("per_example", None)
    return result


def _number(value):
    return "n/a" if value is None else f"{value:+.3f}"


def print_results(results, names):
    """Print pair measurements, cutoff checks and model-level averages."""
    rows = []
    for result in results:
        cover = result["coverage"]
        print(f"\n--- {cover['model_a']} vs {cover['model_b']} ---")
        print(f"{cover['n_selection']} selection, {cover['n_evaluation']} evaluation, "
              f"{cover['n_paired_structure']} ordinary/repetition pairs")
        if cover["status"] != "complete":
            print("Too few aligned passages; comparison skipped.")
            continue
        rows.append(result["summary"])
        bands = result["conditions"]["ordinary"]["result"]["symmetric_bands"]
        print(f"{'depth':<8} {'same':>9} {'random':>9} {'different':>10} {'gap':>9} {'same-diff':>10}")
        for band in ("early", "late"):
            values = bands[band]
            print(f"{band:<8} " + " ".join(f"{_number(values[k]):>9}" for k in (
                "matched_mean", "random_mean", "shuffled_matched_mean",
                "matched_minus_random", "same_minus_shuffled")))
            intervals = values["confidence_intervals"]
            if intervals:
                for metric, label in (("matched_minus_random", "gap"), ("same_minus_shuffled", "same-diff")):
                    lo, hi = intervals[metric]
                    print(f"  {label} 95% passage-bootstrap interval: [{lo:+.3f}, {hi:+.3f}]")

        print("cutoff   early gap   late gap   late same-diff")
        for row in result["sensitivity_summary"]:
            print(f"{row['cutoff']:.1f}      {_number(row['early_gap']):>8}   "
                  f"{_number(row['late_gap']):>8}   {_number(row['late_input_excess']):>14}")
        for name, sink in result["selection_sinks"].items():
            print(f"{name}: excluded-head share {sink['parked_share']:.3f}, "
                  f"BOS mass {sink['mean_bos_mass']:.3f}, first-region mass {sink['mean_first_region_mass']:.3f}")
        change = result["structure_change"]
        if change:
            print("repetition minus ordinary, matched passages:")
            for band in ("early", "late"):
                values = change["bands"][band]["matched_minus_random"]
                interval = values["confidence_interval"]
                ci = f" [{interval[0]:+.3f}, {interval[1]:+.3f}]" if interval else ""
                print(f"  {band}: gap change {_number(values['mean_change'])}{ci}")
        common = result.get("common_cohort_result")
        if common:
            values = common["symmetric_bands"]
            print(f"common passages (n={len(result['common_cohort_ids'])}): "
                  f"early gap {_number(values['early']['matched_minus_random'])}, "
                  f"late gap {_number(values['late']['matched_minus_random'])}")

    summary = summarise(rows, names)
    print(f"\n{len(rows)}/{len(results)} model pairs scored")
    print("model             partners    mean late gap    mean late same-diff")
    for row in summary["per_model"]:
        print(f"{row['model']:<18} {row['n_partners']:>5} "
              f"{_number(row['mean_late_gap']):>16} {_number(row['mean_late_input_excess']):>22}")
    print("\ncorrelations with late gap (overlapping model pairs):")
    for label, key in (("log smaller size", "log10_smaller_size_vs_late_gap"),
                       ("log size ratio", "log10_size_ratio_vs_late_gap"),
                       ("mean excluded-head share", "mean_sink_share_vs_late_gap")):
        row = summary["correlations"][key]
        print(f"{label}: r={_number(row['pearson_r'])}, {row['n_pairs']} pairs")


def plot_summary(path, results):
    """Compare the late-layer contrasts in lower-triangular model matrices."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle

    labels = {"pythia-70m": "Pythia 70M", "pythia-160m": "Pythia 160M",
              "pythia-410m": "Pythia 410M", "pythia-1b": "Pythia 1B",
              "gpt2": "GPT-2", "gpt2-medium": "GPT-2 medium",
              "gpt-neo-125m": "GPT-Neo 125M", "opt-125m": "OPT 125M"}
    metrics = (("late_gap", "Selected matches − random heads"),
               ("late_input_excess", "Same text − different text"))
    index = {name: i for i, name in enumerate(MODELS)}
    matrices = [np.full((len(MODELS), len(MODELS)), np.nan) for _ in metrics]
    counts = np.zeros((len(MODELS), len(MODELS)), dtype=int)
    scored = [r for r in results if r["coverage"]["status"] == "complete"]
    for result in scored:
        row = result["summary"]
        i, j = sorted((index[row["model_a"]], index[row["model_b"]]), reverse=True)
        counts[i, j] = row["n_evaluation"]
        for matrix, (key, _) in zip(matrices, metrics):
            value = _finite(row.get(key))
            if value is not None:
                matrix[i, j] = value

    finite = np.concatenate([m[np.isfinite(m)] for m in matrices])
    limit = max(.05, float(np.max(np.abs(finite))) if len(finite) else .05)
    limit = float(np.ceil(limit * 20) / 20)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    settings = {"font.family": "DejaVu Sans", "font.size": 10,
                "figure.facecolor": "white", "axes.facecolor": "white",
                "text.color": "#243342", "axes.titlecolor": "#243342",
                "axes.labelcolor": "#243342", "axes.grid": False,
                "xtick.color": "#536170", "ytick.color": "#536170",
                "pdf.fonttype": 42, "ps.fonttype": 42}
    with plt.rc_context(settings):
        fig, axes = plt.subplots(1, 2, figsize=(14, 7.8))
        fig.subplots_adjust(left=.11, right=.90, top=.81, bottom=.24, wspace=.42)
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("white")
        names = [labels.get(name, name) for name in MODELS]
        for ax, matrix, (_, title), letter in zip(axes, matrices, metrics, ("a", "b")):
            im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=norm)
            ax.set_xticks(range(len(MODELS)), names, rotation=45,
                          ha="right", rotation_mode="anchor")
            ax.set_yticks(range(len(MODELS)), names)
            ax.tick_params(length=0, labelsize=9, pad=7)
            ax.set_title(f"{letter}  {title}", loc="left", fontsize=12, pad=20)
            for spine in ax.spines.values():
                spine.set_visible(False)
            for i in range(len(MODELS)):
                for j in range(i):
                    value = matrix[i, j]
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1,
                                           facecolor="none" if np.isfinite(value) else "#eef1f4",
                                           edgecolor="white", linewidth=1.7))
                    if np.isfinite(value):
                        rgba = cmap(norm(value))
                        luminance = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
                        ax.text(j, i, f"{value:+.3f}\nn={counts[i, j]}", ha="center", va="center",
                                fontsize=8, color="white" if luminance < .5 else "#243342")
                    else:
                        ax.text(j, i, "—", ha="center", va="center", color="#8a96a2")
        colour_ax = fig.add_axes([.93, .29, .014, .44])
        fig.colorbar(im, cax=colour_ax, label="Cosine similarity difference · shared scale")
        fig.suptitle("E8 · Held-out attention agreement", x=.055, y=.98,
                     ha="left", fontsize=17, fontweight="semibold")
        fig.text(.055, .92, f"Late layers (relative depth ≥ 0.75) · identical token boundaries · {len(scored)}/{len(results)} pairs scored",
                 fontsize=10.5, color="#536170")
        fig.text(.055, .065, "Each pair appears once; both matching directions are averaged. n = evaluation excerpts.",
                 fontsize=9, color="#536170")
        fig.text(.055, .033, "Grey / — = unavailable comparison; blank upper triangle and diagonal are omitted.",
                 fontsize=9, color="#536170")
        path = Path(path)
        fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        plt.close(fig)


def plot_late_contrasts(path, results):
    """Show each model pair once, with its two late-layer contrasts."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    from src.attnlib import style
    style.apply()
    scored = [r for r in results if r["coverage"]["status"] == "complete"]
    labels = {"pythia-70m": "Pythia 70M", "pythia-160m": "Pythia 160M",
              "pythia-410m": "Pythia 410M", "pythia-1b": "Pythia 1B",
              "gpt2": "GPT-2", "gpt2-medium": "GPT-2 medium",
              "gpt-neo-125m": "GPT-Neo 125M", "opt-125m": "OPT 125M"}
    height = max(5.3, 1.9 + .29 * len(scored))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, height), sharey=True,
                             facecolor="white")
    fig.subplots_adjust(left=.32, right=.95, bottom=.85 / height, top=1 - 1.15 / height, wspace=.28)
    fig.text(.035, 1 - .16 / height, "E8 · Held-out attention agreement", fontsize=16, weight="semibold", va="top")
    fig.text(.035, 1 - .52 / height, "Identical token boundaries; sample counts vary by pair" + f"  ·  {len(scored)}/{len(results)} eligible pairs",
             fontsize=10, color="#4b5563", va="top")
    metrics = (("matched_minus_random", "Selected matches − random heads", "#197a72"),
               ("same_minus_shuffled", "Same text − different text", "#345d9d"))
    ranges = [0, .30]
    pair_labels = []
    counts = {r["coverage"]["n_evaluation"] for r in scored}
    for i, result in enumerate(scored):
        cover = result["coverage"]
        a, b = (labels.get(cover[k], cover[k]) for k in ("model_a", "model_b"))
        count = f"   (n={cover['n_evaluation']})" if len(counts) > 1 else ""
        pair_labels.append(f"{a} / {b}" + count)
        band = result["conditions"]["ordinary"]["result"]["symmetric_bands"]["late"]
        for ax, (key, title, color) in zip(axes, metrics):
            value = band.get(key)
            if value is None or not np.isfinite(value):
                ax.text(.015, i, "not estimable", va="center", fontsize=8, color="#6b7280")
                continue
            interval = band.get("confidence_intervals", {}).get(key)
            if interval is not None and np.isfinite(interval).all():
                ax.hlines(i, *interval, color=color, lw=1.5, zorder=3)
                ranges.extend(interval)
            ax.scatter(value, i, s=25, color=color, zorder=4)
            ax.annotate(f"{value:+.3f}", (max(value, interval[1]) if interval else value, i),
                        xytext=(7, 0), textcoords="offset points", va="center",
                        fontsize=8, color="#27313f")
            ranges.append(value)
    low, high = min(ranges), max(ranges)
    low = min(-.012, low - .015)
    high += .065
    for ax, (key, title, color) in zip(axes, metrics):
        ax.set_facecolor("white")
        ax.set_title(title, loc="left", fontsize=11, color="#27313f", pad=15)
        ax.set(xlabel="Cosine similarity difference", xlim=(low, high))
        ax.set_ylim(len(scored) - .4, -.6)
        ax.grid(False)
        ax.xaxis.grid(True, color="#e7ebef", linewidth=.7)
        ax.axvline(0, color="#89929d", linewidth=.9, linestyle=":")
        ax.xaxis.set_major_locator(MultipleLocator(.1))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.tick_params(axis="both", colors="#384454", labelsize=9)
        for spine in ("top", "left", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#cfd6de")
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(scored)), pair_labels)
    axes[0].tick_params(axis="y", pad=12)
    if not scored:
        axes[0].text(.5, .5, "No eligible model pairs", transform=axes[0].transAxes,
                     ha="center", va="center")
    sample_note = f"n = {next(iter(counts))} evaluation excerpts per pair." if len(counts) == 1 else "n = evaluation excerpts."
    fig.text(.035, .33 / height, "Late = relative depth ≥ 0.75; both matching directions are averaged. " + sample_note,
             fontsize=8.5, color="#4b5563")
    fig.text(.035, .12 / height, "Bars: 95% paired-excerpt bootstrap intervals, conditional on selected matches; pairs share models.",
             fontsize=8.5, color="#4b5563")
    path = Path(path)
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_depth(path, results):
    """Plot ordinary-passage scores in the first model's layer order."""
    import matplotlib.pyplot as plt
    from src.attnlib import style
    style.apply()
    scored = [r for r in results if r["coverage"]["status"] == "complete"]
    cols = min(3, max(1, len(scored)))
    nrows = max(1, (len(scored) + cols - 1) // cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(5 * cols, 3.5 * nrows), squeeze=False)
    for ax, result in zip(axes.flat, scored):
        cover = result["coverage"]
        rows = result["conditions"]["ordinary"]["result"]["directions"][0]["rows"]
        for key, label, color in (("matched_mean", "same passage", style.SERIES[0]),
                                  ("shuffled_matched_mean", "different passages", style.SERIES[1]),
                                  ("random_mean", "random heads, same passage", style.MUTED)):
            ax.plot([r["depth"] for r in rows], [r[key] for r in rows], marker="o",
                    markersize=3, color=color, label=label)
        ax.set(title=f"{cover['model_a']} → {cover['model_b']} · n={cover['n_evaluation']}",
               xlabel="relative depth of source model", ylabel="cosine similarity", ylim=(-.03, 1.03))
    for ax in list(axes.flat)[len(scored):]:
        ax.set_visible(False)
    if scored:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(.5, 1))
    else:
        fig.text(.5, .5, "No model pair has enough aligned passages", ha="center")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_factors(path, results):
    """Plot late agreement against size, ratio, sinks and paired repetition."""
    import matplotlib.pyplot as plt
    from src.attnlib import style
    style.apply()
    scored = [r for r in results if r["coverage"]["status"] == "complete"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    predictors = (
        (lambda r: r["smaller_size_m"], "smaller model (M parameters)", "log"),
        (lambda r: r["size_ratio"], "parameter-count ratio", "log"),
        (lambda r: (r["sink_share_a"] + r["sink_share_b"]) / 2, "mean excluded-head share", "linear"),
    )
    for same, label, color in ((True, "same family", style.SERIES[0]),
                                (False, "different families", style.SERIES[1])):
        group = [r for r in scored if r["summary"]["same_family"] == same]
        for ax, (predictor, xlabel, scale) in zip(axes.flat, predictors):
            rows = [r["summary"] for r in group if r["summary"]["late_gap"] is not None]
            ax.scatter([predictor(r) for r in rows], [r["late_gap"] for r in rows],
                       color=color, s=45, label=label, zorder=4,
                       edgecolor=style.PAPER, linewidth=1.2)
            ax.set(xlabel=xlabel, ylabel="late matched-minus-random gap", xscale=scale)
            ax.set_axisbelow(True)
            ax.margins(0.12)
            if same:
                ax.axhline(0, color=style.MUTED, lw=1, ls=":")
        pairs = [r for r in group if r["structure_change"] is not None]
        values = [tuple(r["conditions"][c]["result"]["symmetric_bands"]["late"]["matched_minus_random"]
                        for c in ("ordinary_paired", "repetition")) for r in pairs]
        values = [(a, b) for a, b in values if a is not None and b is not None]
        axes[1, 1].scatter([a for a, _ in values], [b for _, b in values], color=color,
                           s=45, label=label, zorder=4,
                           edgecolor=style.PAPER, linewidth=1.2)
        axes[1, 1].set_axisbelow(True)
    ax = axes[1, 1]
    ax.margins(0.12)
    low = min(ax.get_xlim()[0], ax.get_ylim()[0])
    high = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([low, high], [low, high], color=style.MUTED, lw=1, ls=":")
    ax.set(xlabel="ordinary text: late gap", ylabel="repeated text: late gap",
           xlim=(low, high), ylim=(low, high))
    axes[0, 0].legend()
    fig.text(.5, .01, "Points share models. Repetition uses matched ordinary and repeated passages.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, 1))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    if len(set(MODELS)) < 2 or len(MODELS) != len(set(MODELS)):
        raise ValueError("MODELS must contain at least two distinct models")
    if MIN_EXAMPLES < 2:
        raise ValueError("MIN_EXAMPLES must be at least two")

    manifest = load_passages(SELECTION, EVALUATION, REPETITION, N_TOKENS)
    store = AttentionCache(manifest, CACHE, MODELS)
    Path(OUT).expanduser().mkdir(parents=True, exist_ok=True)

    common_ids, _, _ = store.aligned_ids(MODELS)
    _, common_conditions = split_passages(store, common_ids)
    pairs = list(itertools.combinations(MODELS, 2))
    print(f"{len(MODELS)} models, {len(pairs)} candidate pairs", flush=True)
    print(f"sink cutoff {SINK_CUTOFF}; minimum {MIN_EXAMPLES} passages per split", flush=True)

    results = []
    for name_a, name_b in pairs:
        print(f"{name_a} vs {name_b}", flush=True)
        results.append(compare(name_a, name_b, store, common_conditions["ordinary"]))

    print_results(results, MODELS)
    plot_summary(Path(OUT).expanduser() / "summary.png", results)
    plot_late_contrasts(Path(OUT).expanduser() / "late_contrasts.png", results)
    plot_depth(Path(OUT).expanduser() / "e8_depth.png", results)
    plot_factors(Path(OUT).expanduser() / "e8_factors.png", results)
    return results


if __name__ == "__main__":
    main()
