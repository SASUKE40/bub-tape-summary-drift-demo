from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from .experiment import load_result, load_scenario, run_experiment
from .report import render

app = typer.Typer(no_args_is_help=True, help="Measure summary drift across Bub tape handoffs.")
ROOT = Path(__file__).resolve().parents[2]


@app.command()
def run(
    model: Annotated[str, typer.Option(help="Bub/any-llm model, provider:model")] = "openai:gpt-4.1-mini",
    judge_model: Annotated[
        str | None, typer.Option(help="Independent judge model; defaults to --model")
    ] = None,
    rounds: Annotated[int, typer.Option(min=1, max=30)] = 6,
    scenario: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = ROOT / "data/scenario.json",
    tape_dir: Annotated[Path, typer.Option()] = ROOT / ".demo-tapes",
) -> None:
    """Run recursive and source-grounded handoff strategies."""
    result = asyncio.run(
        run_experiment(
            scenario=load_scenario(scenario),
            model=model,
            judge_model=judge_model or model,
            rounds=rounds,
            tape_dir=tape_dir,
        )
    )
    render(result)


@app.command()
def report(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)] = ROOT / ".demo-tapes/report.json",
) -> None:
    """Render an existing report without calling a model."""
    render(load_result(path))


if __name__ == "__main__":
    app()
