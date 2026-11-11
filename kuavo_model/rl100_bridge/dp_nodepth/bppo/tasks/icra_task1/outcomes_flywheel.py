"""Create mixed success/failure labels from an audited bag-to-episode mapping."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from kuavo_data.flywheel.inventory import classify
from .release_terminal import (
    CSV_FIELDS, FAILURE_LABEL_SEMANTICS, build_release_terminal_records,
)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout-as-terminal", action="store_true",
                        help="Treat validated time-limit failures as task terminals with reward 0")
    args = parser.parse_args()
    root = Path(args.dataset_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    audit_path = Path(str(output) + ".exclusions.json")
    if output.exists() or audit_path.exists():
        raise FileExistsError("Annotation or exclusion audit already exists")
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    count = int(info["total_episodes"])
    mapping_path = Path(args.mapping).expanduser().resolve()
    with mapping_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"episode_index", "source", "bag_path", "metadata_path"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Mapping requires columns {sorted(required)}")
        rows = list(reader)
    mapping = {}
    seen_bags = set()
    failures = set()
    for row in rows:
        episode = int(row["episode_index"])
        if episode in mapping or not 0 <= episode < count:
            raise ValueError(f"Invalid or duplicate episode index {episode}")
        bag = Path(row["bag_path"]).expanduser().resolve(strict=True)
        if not bag.is_file() or bag.stat().st_size == 0 or bag in seen_bags:
            raise ValueError(f"Missing, empty or duplicate bag: {bag}")
        seen_bags.add(bag)
        if row["source"] == "expert_success":
            if row["metadata_path"].strip():
                raise ValueError("expert_success must not contain flywheel metadata")
            outcome = "expert_success"
        elif row["source"] == "flywheel":
            metadata_path = Path(row["metadata_path"]).expanduser().resolve(strict=True)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if Path(metadata["bag"]).expanduser().resolve() != bag:
                raise ValueError(f"Metadata bag mismatch: {metadata_path}")
            outcome = classify(metadata)
            if outcome == "invalid":
                raise ValueError(f"Unsafe or unclassified collection: {metadata_path}")
            if outcome == "timeout":
                if not args.timeout_as_terminal:
                    raise ValueError("Timeout failures require explicit --timeout-as-terminal")
                failures.add(episode)
        else:
            raise ValueError(f"Unknown source {row['source']!r}")
        mapping[episode] = (bag, outcome)
    if set(mapping) != set(range(count)):
        raise ValueError("Mapping must cover each dataset episode exactly once")

    records, _ = build_release_terminal_records(
        root, state_dim=16, action_dim=16, left_gripper_index=7,
        right_gripper_index=15, stable_open_frames=3, excluded_episodes=failures,
    )
    import pyarrow.parquet as pq
    lengths = {}
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "frame_index"])
        for episode, frame in zip(table["episode_index"].to_pylist(),
                                  table["frame_index"].to_pylist(), strict=True):
            episode, frame = int(episode), int(frame)
            lengths[episode] = max(lengths.get(episode, 0), frame + 1)
    for episode in sorted(failures):
        records.append({"episode_index": episode, "episode_length": lengths[episode],
                        "left_command_open_frame": "", "left_stable_open_frame": "",
                        "right_command_open_frame": "", "right_stable_open_frame": "",
                        "terminal_frame": lengths[episode] - 1, "terminal_reward": 0.0,
                        "label_semantics": FAILURE_LABEL_SEMANTICS})
    records.sort(key=lambda row: row["episode_index"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    with audit_path.open("x", encoding="utf-8") as stream:
        json.dump({"dataset_total_episodes": count, "labeled_episodes": len(records),
                   "excluded_episode_indices": [], "reason": "",
                   "timeout_as_terminal": bool(args.timeout_as_terminal),
                   "mapping": str(mapping_path)}, stream, indent=2)
        stream.write("\n")
    for episode, (bag, outcome) in sorted(mapping.items()):
        print(f"episode={episode} outcome={outcome} bag={bag}")
    print(f"MIXED_REWARD_ANNOTATIONS_READY: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
