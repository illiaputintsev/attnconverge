"""Shared plot appearance."""

import matplotlib as mpl

INK = "#1a1a1a"
MUTED = "#8a8578"
GRID = "#e4e1d8"
PAPER = "#faf9f6"

# one colour per model family
FAMILY_COLOUR = {
    "Pythia": "#1d9e75",
    "GPT-2": "#d8722c",
    "GPT-Neo": "#7f77dd",
    "OPT": "#c4453a",
}

SERIES = ["#1d9e75", "#d8722c", "#7f77dd", "#c4453a", "#2b6cb0", "#8a8578"]
HEATMAP = "viridis"


def apply():
    mpl.rcParams.update({
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "font.family": "sans-serif",
        "font.sans-serif": ["IBM Plex Sans", "DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "axes.edgecolor": GRID,
        "axes.labelsize": 9.5,
        "axes.titlesize": 11.5,
        "axes.titleweight": "medium",
        "axes.titlepad": 12,
        "axes.labelpad": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "lines.solid_capstyle": "round",
        "figure.dpi": 130,
    })


def titled(ax, title, subtitle=None):
    if subtitle:
        ax.set_title(title, loc="left", pad=26)
        ax.text(0, 1.025, subtitle, transform=ax.transAxes,
                fontsize=8.8, color=MUTED, va="bottom")
    else:
        ax.set_title(title, loc="left", pad=12)