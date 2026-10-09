#!/usr/bin/env python
"""Shared publication plotting helpers for the metacognition experiments.

The defaults follow the compact visual conventions commonly used by Nature
Machine Intelligence: sans-serif type, thin axes, restrained color, explicit
uncertainty, and vector output alongside a high-resolution PNG preview.
"""

import math
import os
from typing import Iterable, Optional, Tuple


SINGLE_COLUMN_IN = 3.50
DOUBLE_COLUMN_IN = 7.20

# Okabe-Ito, with neutral tones added for references and inactive targets.
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#202124",
    "gray": "#7A7F87",
    "light_gray": "#D9DDE3",
}

CLUSTER_COLORS = [
    COLORS["blue"],
    COLORS["vermillion"],
    COLORS["green"],
    COLORS["purple"],
    COLORS["orange"],
    COLORS["sky"],
]


def apply_nmi_style() -> None:
    """Apply a deterministic, journal-oriented Matplotlib style."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "axes.prop_cycle": mpl.cycler(
                color=[
                    COLORS["blue"],
                    COLORS["vermillion"],
                    COLORS["green"],
                    COLORS["orange"],
                    COLORS["purple"],
                    COLORS["sky"],
                ]
            ),
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "Noto Sans CJK SC",
                "DejaVu Sans",
            ],
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "axes.titleweight": "semibold",
            "axes.linewidth": 0.7,
            "axes.edgecolor": COLORS["black"],
            "axes.labelcolor": COLORS["black"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.fontsize": 7.0,
            "legend.frameon": False,
            "lines.linewidth": 1.35,
            "lines.markersize": 4.0,
            "patch.linewidth": 0.6,
            "figure.dpi": 120,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def style_axis(axis, grid: Optional[str] = "y") -> None:
    """Apply subtle guides without enclosing the data in a heavy frame."""
    axis.spines["left"].set_linewidth(0.7)
    axis.spines["bottom"].set_linewidth(0.7)
    if grid:
        axis.grid(
            True,
            axis=grid,
            color=COLORS["light_gray"],
            linewidth=0.45,
            alpha=0.65,
            zorder=0,
        )
        axis.set_axisbelow(True)


def panel_label(axis, label: str, x: float = -0.14, y: float = 1.06) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


def save_figure(figure, path: str, dpi: int = 450, vector: bool = True) -> None:
    """Save a PNG preview and, by default, an editable vector PDF."""
    root, extension = os.path.splitext(path)
    if extension.lower() not in {".png", ".pdf", ".svg"}:
        root = path
    os.makedirs(os.path.dirname(root) or ".", exist_ok=True)
    figure.savefig(f"{root}.png", dpi=dpi, facecolor="white")
    if vector:
        figure.savefig(f"{root}.pdf", facecolor="white")


def symmetric_limit(values: Iterable[float], floor: float = 1e-6) -> float:
    finite = [abs(float(value)) for value in values if math.isfinite(float(value))]
    return max(max(finite, default=floor), floor)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(probability * (1.0 - probability) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def metric_label(name: str) -> str:
    labels = {
        "generated_tokens": "Generated tokens",
        "style_line_count": "Lines",
        "style_paragraph_count": "Paragraphs",
        "style_bullet_line_count": "Bullet lines",
        "style_heading_count": "Headings",
        "style_code_block_count": "Code blocks",
        "style_question_count": "Questions",
        "style_explanation_marker_count": "Explanation markers",
        "style_example_marker_count": "Example markers",
        "style_self_correction_marker_count": "Self-corrections",
        "style_hedging_score": "Hedging markers",
        "style_certainty_score": "Certainty markers",
        "style_politeness_score": "Politeness markers",
        "style_first_person_count": "First-person terms",
        "style_second_person_count": "Second-person terms",
    }
    return labels.get(name, name.replace("style_", "").replace("_", " ").capitalize())


def cluster_color(cluster: int) -> str:
    return CLUSTER_COLORS[int(cluster) % len(CLUSTER_COLORS)]
