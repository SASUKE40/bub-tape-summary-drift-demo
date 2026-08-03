from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from bub.builtin.store import FileTapeStore
from bub.builtin.tape import Tape
from bub.tape import AsyncTapeStoreAdapter, TapeContext, TapeEntry

from .llm import LLM
from .models import (
    ClaimStatus,
    ExperimentResult,
    Fact,
    Grade,
    RoundResult,
    Scenario,
    StrategyName,
    SummaryPayload,
)

SUMMARY_SYSTEM = """You create phase-handoff summaries for a long-running software project.
Compress aggressively into concise prose. Preserve details that appear operationally important.
Do not invent facts. If source IDs are supplied, return the IDs you relied on.
"""

JUDGE_SYSTEM = """You audit a handoff summary against atomic source facts.
For every fact, choose exactly one status:
- preserved: the summary explicitly retains the material meaning,
  including numbers, negations, scope, and uncertainty;
- contradicted: the summary changes or negates material meaning, scope, certainty, actor, quantity, or timing;
- missing: the material meaning is absent or only referenced vaguely.
A generic phrase such as "constraints remain unchanged" does NOT preserve any underlying fact.
A preserved fact needs an explicit quote containing its material details; otherwise mark it missing.
List claims in the summary that are unsupported by the facts or update.
"""


@dataclass
class StrategyRun:
    name: StrategyName
    tape: Tape
    latest_summary: str = ""


def load_scenario(path: Path) -> Scenario:
    return Scenario.model_validate_json(path.read_text(encoding="utf-8"))


def facts_text(facts: Iterable[Fact]) -> str:
    return "\n".join(f"[{fact.id}] ({fact.category}) {fact.statement}" for fact in facts)


def keyword_recall(summary: str, facts: list[Fact]) -> float:
    """Transparent lexical baseline; semantic grading comes from the judge model."""
    text = _normalize(summary)
    kept = 0
    for fact in facts:
        if all(_normalize(keyword) in text for keyword in fact.keywords):
            kept += 1
    return kept / len(facts) if facts else 1.0


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


async def summarize_recursive(llm: LLM, previous: str, update: str, fact_ids: list[str]) -> SummaryPayload:
    user = f"""Create the next compact handoff from ONLY the previous handoff and current update.
This intentionally models recursive summarization; do not retrieve omitted older evidence.

Previous handoff:
{previous}

Current update:
{update}

Known source ID vocabulary (use only IDs directly supported by the text above):
{', '.join(fact_ids)}
"""
    return await llm.json(system=SUMMARY_SYSTEM, user=user, schema=SummaryPayload)


async def summarize_grounded(llm: LLM, facts: list[Fact], updates: list[str]) -> SummaryPayload:
    user = f"""Create a compact handoff from the ORIGINAL atomic facts and all updates.
Preserve hard constraints, exceptions, ownership, scope, uncertainty, rollback triggers, and exact quantities.
Each retained claim must be traceable to source IDs. Return all fact IDs materially represented.

Original facts:
{facts_text(facts)}

Updates so far:
{chr(10).join(f'- U{i + 1}: {value}' for i, value in enumerate(updates))}
"""
    return await llm.json(system=SUMMARY_SYSTEM, user=user, schema=SummaryPayload)


async def grade_summary(judge: LLM, facts: list[Fact], updates: list[str], summary: str) -> Grade:
    user = f"""Audit this summary.

Atomic facts:
{facts_text(facts)}

Updates through the current round:
{chr(10).join(f'- U{i + 1}: {value}' for i, value in enumerate(updates))}

Summary under audit:
{summary}

Return one assessment for every fact ID, in the same order.
"""
    grade = await judge.json(system=JUDGE_SYSTEM, user=user, schema=Grade)
    by_id = {item.fact_id: item for item in grade.assessments}
    missing_ids = [fact.id for fact in facts if fact.id not in by_id]
    if missing_ids:
        raise ValueError(f"judge omitted fact assessments: {missing_ids}")
    grade.assessments = [by_id[fact.id] for fact in facts]
    return grade


async def _append_source(tape: Tape, scenario: Scenario) -> None:
    await tape.ensure_bootstrap_anchor()
    await tape.store.append(
        tape.name,
        TapeEntry.event(
            "source_facts",
            {"scenario": scenario.scenario, "facts": [fact.model_dump() for fact in scenario.facts]},
        ),
    )


async def run_experiment(
    *,
    scenario: Scenario,
    model: str,
    judge_model: str,
    rounds: int,
    tape_dir: Path,
) -> ExperimentResult:
    tape_dir.mkdir(parents=True, exist_ok=True)
    store = AsyncTapeStoreAdapter(FileTapeStore(tape_dir))
    base_tape = Tape(archive_path=tape_dir / "archive", store=store, context=TapeContext(anchor=None))
    runs = {
        StrategyName.recursive: StrategyRun(
            StrategyName.recursive, base_tape.scoped("summary_drift__recursive")
        ),
        StrategyName.grounded: StrategyRun(
            StrategyName.grounded, base_tape.scoped("summary_drift__grounded")
        ),
    }
    for run in runs.values():
        await run.tape.reset()
        await _append_source(run.tape, scenario)

    summarizer = LLM(model)
    judge = LLM(judge_model)
    results: list[RoundResult] = []
    updates: list[str] = []
    selected_updates = [
        scenario.round_updates[index % len(scenario.round_updates)] for index in range(rounds)
    ]

    initial = facts_text(scenario.facts)
    runs[StrategyName.recursive].latest_summary = initial

    for round_number, update in enumerate(selected_updates, start=1):
        updates.append(update)
        for strategy in (StrategyName.recursive, StrategyName.grounded):
            run = runs[strategy]
            await run.tape.store.append(
                run.tape.name,
                TapeEntry.message({"role": "user", "content": update}, round=round_number),
            )
            if strategy is StrategyName.recursive:
                payload = await summarize_recursive(
                    summarizer,
                    run.latest_summary,
                    update,
                    [fact.id for fact in scenario.facts],
                )
            else:
                payload = await summarize_grounded(summarizer, scenario.facts, updates)

            run.latest_summary = payload.summary
            anchor = f"round/{round_number}"
            await run.tape.handoff(
                name=anchor,
                state={
                    "strategy": strategy.value,
                    "summary": payload.summary,
                    "source_ids": payload.source_ids,
                    "round": round_number,
                },
            )
            grade = await grade_summary(judge, scenario.facts, updates, payload.summary)
            preserved = sum(item.status is ClaimStatus.preserved for item in grade.assessments)
            contradicted = sum(item.status is ClaimStatus.contradicted for item in grade.assessments)
            results.append(
                RoundResult(
                    strategy=strategy,
                    round=round_number,
                    anchor=anchor,
                    summary=payload.summary,
                    source_ids=payload.source_ids,
                    grade=grade,
                    deterministic_recall=keyword_recall(payload.summary, scenario.facts),
                    semantic_recall=preserved / len(scenario.facts),
                    contradicted=contradicted,
                    unsupported=len(grade.unsupported_claims),
                    summary_chars=len(payload.summary),
                )
            )
            await run.tape.append_event("drift_grade", results[-1].model_dump(mode="json"))

    experiment = ExperimentResult(
        model=model,
        judge_model=judge_model,
        rounds=rounds,
        scenario=scenario.scenario,
        results=results,
        tape_dir=str(tape_dir.resolve()),
    )
    (tape_dir / "report.json").write_text(experiment.model_dump_json(indent=2), encoding="utf-8")
    return experiment


def load_result(path: Path) -> ExperimentResult:
    return ExperimentResult.model_validate(json.loads(path.read_text(encoding="utf-8")))
