"""E10: Attention correspondence across tokenisers

Each passage is divided into 16 spans with boundaries shared by all ten
models. Attention is measured from the final token of each span, summing key
weights within spans. All model pairs use the same passages and boundaries.

Head matches, random targets and sink exclusions are fixed on the selection
set, then evaluated on ordinary and repeated passages. Span aggregation removes
within-span routing; sink mass includes BOS and the first span.
"""

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np

from src.attnlib.matching import evaluate_matches, fit_matches, stable_seed
from src.attnlib.extract import MODEL_INFO, AttentionCache, load_passages, split_passages
from src.attnlib.metrics import paired_structure_difference, sink_summary
from src.attnlib import style


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
OUT = "results/e10"

MIN_EXAMPLES = 25
SEED = 20260911
SINK_CUTOFF = 0.9
SENSITIVITY_CUTOFFS = (0.7, 1.0)
N_SHUFFLES = 100  # supplementary control; the different-input mean uses all pairs
N_BOOTSTRAPS = 300
N_SPANS = 16


def score_conditions(name_a, name_b, store, conditions, spans, plans, seed):
    """Score frozen head pairs on ordinary and repeated passages."""
    scores, sensitivity = {}, []
    for condition, ids in conditions.items():
        if len(ids) < MIN_EXAMPLES:
            scores[condition] = {"status": "deferred_insufficient_aligned_examples", "ids": ids}
            continue

        a, b = store.load(name_a, ids, spans), store.load(name_b, ids, spans)
        result = evaluate_matches(
            plans[str(SINK_CUTOFF)], a["attention"], b["attention"],
            seed=seed + 1, n_shuffles=N_SHUFFLES, n_bootstrap=N_BOOTSTRAPS,
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


def compare(name_a, name_b, store, selection_ids, conditions, rejected, spans):
    """Match and evaluate one model pair on the common span partition."""
    coverage = {
        "model_a": name_a, "model_b": name_b,
        "n_selection": len(selection_ids), "n_evaluation": len(conditions["ordinary"]),
        "n_paired_structure": len(conditions["repetition"]), "n_rejected": len(rejected),
        "alignment": "spans", "status": "deferred_insufficient_aligned_examples",
    }
    if len(selection_ids) < MIN_EXAMPLES or len(conditions["ordinary"]) < MIN_EXAMPLES:
        return {"coverage": coverage, "selection_ids": selection_ids,
                "conditions": conditions, "rejected_ids": rejected}

    # Selection uses the same span representation as evaluation.
    selected_a = store.load(name_a, selection_ids, spans)
    selected_b = store.load(name_b, selection_ids, spans)
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

    scores, sensitivity = score_conditions(name_a, name_b, store, conditions, spans, plans, seed)
    structure = None
    if all(scores[c]["status"] == "complete" for c in ("ordinary_paired", "repetition")):
        structure = paired_structure_difference(
            scores["ordinary_paired"]["result"], scores["repetition"]["result"],
            seed=seed + 2, n_bootstrap=N_BOOTSTRAPS,
        )

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
        "notes": [
            "Pairs share models and are not independent replications; per-model means and correlations are descriptive only.",
            "Size, ratio, architecture, training and sink prevalence can covary. These summaries do not identify mechanisms or isolate scale effects.",
            "Small or absent correlations do not establish equivalence, scale independence or absence of an effect. No p-values are reported.",
            "Each correlation reports its own complete-pair count; different summaries may use different subsets.",
        ],
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
    """Show each model pair once, with its two late-layer contrasts."""
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
    fig.text(.035, 1 - .16 / height, "E10 · Attention agreement across tokenisers", fontsize=16, weight="semibold", va="top")
    fig.text(.035, 1 - .52 / height, f"Common text spans across {len(MODELS)} models" + f"  ·  {len(scored)}/{len(results)} eligible pairs",
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
                       color=color, s=45, label=label)
            ax.set(xlabel=xlabel, ylabel="late matched-minus-random gap", xscale=scale)
            if same:
                ax.axhline(0, color=style.MUTED, lw=1, ls=":")
        pairs = [r for r in group if r["structure_change"] is not None]
        values = [tuple(r["conditions"][c]["result"]["symmetric_bands"]["late"]["matched_minus_random"]
                        for c in ("ordinary_paired", "repetition")) for r in pairs]
        values = [(a, b) for a, b in values if a is not None and b is not None]
        axes[1, 1].scatter([a for a, _ in values], [b for _, b in values], color=color, s=45, label=label)
    ax = axes[1, 1]
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
    accepted, rejected, spans = store.aligned_ids(MODELS, "spans", N_SPANS)
    selection_ids, conditions = split_passages(store, accepted)
    pairs = list(itertools.combinations(MODELS, 2))
    print(f"{len(MODELS)} models, {len(pairs)} pairs, {N_SPANS} shared spans")
    print(f"Common passages: {len(selection_ids)} selection, "
          f"{len(conditions['ordinary'])} evaluation, "
          f"{len(conditions['repetition'])} ordinary/repetition pairs", flush=True)

    results = []
    for name_a, name_b in pairs:
        result = compare(name_a, name_b, store, selection_ids, conditions, rejected, spans)
        results.append(result)
        cover = result["coverage"]
        if cover["status"] != "complete":
            print(f"{name_a} vs {name_b}: skipped (minimum {MIN_EXAMPLES} passages per split)",
                  flush=True)
            continue
        print(f"{name_a} vs {name_b}: complete", flush=True)

    print_results(results, MODELS)
    output = Path(OUT).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    plot_summary(output / "summary.png", results)
    plot_depth(output / "e10_depth.png", results)
    plot_factors(output / "e10_factors.png", results)
    print(f"\nFigures saved to {output}")
    return results


if __name__ == "__main__":
    main()
