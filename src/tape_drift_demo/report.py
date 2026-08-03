from __future__ import annotations

from collections import defaultdict

from rich.console import Console
from rich.table import Table

from .models import ExperimentResult, StrategyName


def render(result: ExperimentResult, console: Console | None = None) -> None:
    console = console or Console()
    console.print(f"\n[bold]Tape summary-drift experiment[/bold] — {result.scenario}")
    console.print(f"summarizer={result.model}  judge={result.judge_model}  rounds={result.rounds}")

    table = Table(title="Retention by handoff round")
    table.add_column("Strategy")
    table.add_column("Round", justify="right")
    table.add_column("Semantic recall", justify="right")
    table.add_column("Exact-key recall", justify="right")
    table.add_column("Contradictions", justify="right")
    table.add_column("Unsupported", justify="right")
    table.add_column("Chars", justify="right")

    grouped: dict[StrategyName, list] = defaultdict(list)
    for row in result.results:
        grouped[row.strategy].append(row)
        table.add_row(
            row.strategy.value,
            str(row.round),
            f"{row.semantic_recall:.0%}",
            f"{row.deterministic_recall:.0%}",
            str(row.contradicted),
            str(row.unsupported),
            str(row.summary_chars),
        )
    console.print(table)

    comparison = Table(title="First → last handoff")
    comparison.add_column("Strategy")
    comparison.add_column("Semantic recall")
    comparison.add_column("Change")
    comparison.add_column("Last missing/contradicted IDs")
    for strategy in (StrategyName.recursive, StrategyName.grounded):
        rows = sorted(grouped[strategy], key=lambda item: item.round)
        first, last = rows[0], rows[-1]
        lost = [
            item.fact_id
            for item in last.grade.assessments
            if item.status.value != "preserved"
        ]
        comparison.add_row(
            strategy.value,
            f"{first.semantic_recall:.0%} → {last.semantic_recall:.0%}",
            f"{last.semantic_recall - first.semantic_recall:+.0%}",
            ", ".join(lost) or "none",
        )
    console.print(comparison)
    console.print(f"\nRaw Bub tapes and machine-readable report: [cyan]{result.tape_dir}[/cyan]")
