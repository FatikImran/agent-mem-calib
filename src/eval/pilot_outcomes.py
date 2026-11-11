"""Summarize saved Phase 1 pilot outcomes without loading a model."""

import json
from pathlib import Path

from src.eval.parameter_f1 import compute_f1


def summarize(records):
    parsed = [row for row in records if row["parsed"]]
    wrong_arguments = 0
    for row in parsed:
        actual = row["prediction"]["arguments"]
        expected = row["gold_arguments"]
        if actual.keys() != expected.keys() or any(
            str(actual[key]).lower() != str(value).lower()
            for key, value in expected.items()
        ):
            wrong_arguments += 1

    def metrics(rows):
        if not rows:
            return None
        scores = compute_f1(
            [row["prediction"] or {} for row in rows],
            [{"arguments": row["gold_arguments"]} for row in rows],
        )
        return dict(zip(("precision", "recall", "f1"), scores))

    return {
        "attempted": len(records),
        "unparsed": len(records) - len(parsed),
        "parsed_exact_arguments": len(parsed) - wrong_arguments,
        "parsed_wrong_arguments": wrong_arguments,
        "all_tasks": metrics(records),
        "parsed_calls_only": metrics(parsed),
    }


if __name__ == "__main__":
    path = Path("results/phase1-pilot.json")
    print(json.dumps(summarize(json.loads(path.read_text(encoding="utf-8"))), indent=2))
