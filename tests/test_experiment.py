from tape_drift_demo.experiment import keyword_recall
from tape_drift_demo.models import Fact


def test_keyword_recall_requires_all_material_tokens() -> None:
    facts = [
        Fact(
            id="F1",
            category="constraint",
            statement="Maximum downtime is 90 seconds.",
            keywords=["downtime", "90 seconds"],
        ),
        Fact(
            id="F2",
            category="owner",
            statement="Mina owns migration.",
            keywords=["mina", "migration"],
        ),
    ]
    assert keyword_recall("Mina owns migration. Downtime must be short.", facts) == 0.5
    assert keyword_recall("Mina owns migration. Downtime is 90 seconds.", facts) == 1.0


def test_keyword_recall_is_case_and_whitespace_insensitive() -> None:
    facts = [
        Fact(
            id="F1",
            category="constraint",
            statement="p95 below 120 ms",
            keywords=["P95", "120   MS"],
        )
    ]
    assert keyword_recall("Keep p95 below 120 ms.", facts) == 1.0
