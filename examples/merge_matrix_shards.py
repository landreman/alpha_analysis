"""Merge milestone 10.3 matrix shard checkpoints into one report JSON.

Shards must share identical controls (the same code and matrix-wide ladders);
disagreeing controls are a provenance error, not something to paper over.

Run from the repository root with::

    python examples/merge_matrix_shards.py /tmp/m/shard*.json \
        --output docs/validation/milestone10.3-real-equilibria.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    merged = None
    for shard in args.shards:
        payload = json.loads(shard.read_text())
        if merged is None:
            merged = {
                "controls": payload["controls"],
                "files": {},
                "cases": {},
            }
        elif payload["controls"] != merged["controls"]:
            raise ValueError(f"{shard} controls disagree with the first shard")
        overlap = set(payload["cases"]) & set(merged["cases"])
        for key in sorted(overlap):
            if payload["cases"][key] != merged["cases"][key]:
                raise ValueError(f"{shard} disagrees on case {key}")
        merged["files"].update(payload["files"])
        merged["cases"].update(payload["cases"])
    classifications = Counter(
        case.get("classification", case.get("outcome"))
        for case in merged["cases"].values()
    )
    failure_classes = Counter()
    for case in merged["cases"].values():
        for name, count in (case.get("failure_class_counts") or {}).items():
            failure_classes[name] += count
    merged["summary"] = {
        "case_count": len(merged["cases"]),
        "classifications": dict(classifications),
        "resolved_fraction": (
            classifications.get("resolved", 0)
            + classifications.get("no_transitions", 0)
        )
        / max(len(merged["cases"]), 1),
        "failure_class_counts": dict(failure_classes),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    print(json.dumps(merged["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
