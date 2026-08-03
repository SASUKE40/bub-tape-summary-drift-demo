from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .models import ClaimStatus, ExperimentResult, StrategyName

COLORS = {
    StrategyName.recursive: "#ef4444",
    StrategyName.grounded: "#22c55e",
}
LABELS = {
    StrategyName.recursive: "Recursive handoff",
    StrategyName.grounded: "Source-grounded handoff",
}


def _group(result: ExperimentResult) -> dict[StrategyName, list]:
    grouped: dict[StrategyName, list] = defaultdict(list)
    for row in result.results:
        grouped[row.strategy].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: item.round)
    return grouped


def _style_axis(axis: Axes) -> None:
    axis.grid(axis="y", alpha=0.18, linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)


def plot_result(result: ExperimentResult, output: Path) -> Path:
    grouped = _group(result)
    figure = plt.figure(figsize=(14, 9), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1, 1.05))
    recall_axis = figure.add_subplot(grid[0, 0])
    detail_axis = figure.add_subplot(grid[0, 1])
    retention_axis = figure.add_subplot(grid[1, :])

    _plot_semantic_recall(recall_axis, grouped)
    _plot_exact_recall(detail_axis, grouped)
    _plot_final_retention(retention_axis, grouped)

    figure.suptitle(
        f"Bub Tape Summary Drift — {result.scenario}\n"
        f"summarizer: {result.model}  •  judge: {result.judge_model}",
        fontsize=16,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def _plot_semantic_recall(axis: Axes, grouped: dict[StrategyName, list]) -> None:
    for strategy in (StrategyName.recursive, StrategyName.grounded):
        rows = grouped[strategy]
        axis.plot(
            [row.round for row in rows],
            [row.semantic_recall * 100 for row in rows],
            marker="o",
            linewidth=2.5,
            markersize=6,
            color=COLORS[strategy],
            label=LABELS[strategy],
        )
    axis.set_title("Semantic fact recall")
    axis.set_xlabel("Handoff round")
    axis.set_ylabel("Preserved facts (%)")
    axis.set_ylim(0, 105)
    axis.set_xticks([row.round for row in grouped[StrategyName.recursive]])
    axis.legend(frameon=False, loc="lower right")
    _style_axis(axis)


def _plot_exact_recall(axis: Axes, grouped: dict[StrategyName, list]) -> None:
    for strategy in (StrategyName.recursive, StrategyName.grounded):
        rows = grouped[strategy]
        axis.plot(
            [row.round for row in rows],
            [row.deterministic_recall * 100 for row in rows],
            marker="s",
            linewidth=2.5,
            markersize=5.5,
            color=COLORS[strategy],
            label=LABELS[strategy],
        )
    axis.set_title("Exact-key detail recall")
    axis.set_xlabel("Handoff round")
    axis.set_ylabel("Facts retaining every key detail (%)")
    axis.set_ylim(0, 105)
    axis.set_xticks([row.round for row in grouped[StrategyName.recursive]])
    axis.legend(frameon=False, loc="lower right")
    _style_axis(axis)


def _plot_final_retention(axis: Axes, grouped: dict[StrategyName, list]) -> None:
    recursive = grouped[StrategyName.recursive][-1]
    grounded = grouped[StrategyName.grounded][-1]
    fact_ids = [item.fact_id for item in recursive.grade.assessments]
    recursive_by_id = {item.fact_id: item.status for item in recursive.grade.assessments}
    grounded_by_id = {item.fact_id: item.status for item in grounded.grade.assessments}

    y_positions = list(range(len(fact_ids)))
    for offset, strategy, statuses in (
        (-0.16, StrategyName.recursive, recursive_by_id),
        (0.16, StrategyName.grounded, grounded_by_id),
    ):
        values = [1 if statuses[fact_id] is ClaimStatus.preserved else 0 for fact_id in fact_ids]
        axis.scatter(
            values,
            [position + offset for position in y_positions],
            s=95,
            color=COLORS[strategy],
            marker="o" if strategy is StrategyName.grounded else "X",
            label=LABELS[strategy],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )

    axis.set_title(f"Final handoff fact retention (round {recursive.round})")
    axis.set_xlabel("Status")
    axis.set_xticks([0, 1], labels=["Missing / contradicted", "Preserved"])
    axis.set_yticks(y_positions, labels=fact_ids)
    axis.set_xlim(-0.2, 1.2)
    axis.invert_yaxis()
    axis.legend(frameon=False, ncols=2, loc="lower center", bbox_to_anchor=(0.5, -0.24))
    _style_axis(axis)


def plot_compact(result: ExperimentResult, output: Path) -> Path:
    """Small chart suitable for README embedding and chat previews."""
    grouped = _group(result)
    figure: Figure
    figure, axis = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    _plot_semantic_recall(axis, grouped)
    axis.set_title("Summary drift across Bub tape handoffs", fontsize=15, fontweight="bold")
    axis.text(
        0.01,
        0.02,
        "Recursive summaries lose source facts; grounded summaries rebuild from original tape evidence.",
        transform=axis.transAxes,
        fontsize=9,
        color="#475569",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output
