"""List simulator flywheel outcomes without ROS or training dependencies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def classify(metadata: dict) -> str:
    inspection = metadata.get("bag_inspection") or {}
    if (metadata.get("rosbag_exit_code") != 0 or metadata.get("error")
            or metadata.get("recording_error") or metadata.get("cleanup_error")
            or metadata.get("bag_error") or not inspection.get("counts")
            or inspection.get("missing_required_topics") is None
            or inspection.get("missing_required_topics")
            or not inspection.get("types")
            or inspection.get("type_mismatches") is None
            or inspection.get("type_mismatches")):
        return "invalid"
    if (metadata.get("success") is True and metadata.get("stop_reason") == "success"
            and metadata.get("candidate_for_review") is True):
        return "success"
    if metadata.get("success") is False and metadata.get("stop_reason") == "time_limit":
        return "timeout"
    return "invalid"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="Flywheel run directories")
    args = parser.parse_args()
    for run_arg in args.runs:
        run = Path(run_arg).expanduser().resolve(strict=True)
        files = sorted((run / "metadata").glob("episode_*.json"))
        if not files:
            raise ValueError(f"No episode metadata found: {run}")
        for path in files:
            meta = json.loads(path.read_text(encoding="utf-8"))
            bag = Path(meta.get("bag", ""))
            outcome = classify(meta)
            if not bag.is_file() or bag.stat().st_size == 0:
                outcome = "invalid"
            print(json.dumps({"outcome": outcome, "episode": meta.get("episode"),
                              "stop_reason": meta.get("stop_reason"),
                              "bag": str(bag), "metadata": str(path)},
                             ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
