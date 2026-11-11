"""Build an episode mapping from converter logs and a source CSV."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion-log", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    log = Path(args.conversion_log).read_text(encoding="utf-8")
    if re.search(r"Error processing|failed bags written|Traceback \(most recent call last\)", log):
        raise ValueError("Conversion log contains failures")
    with Path(args.sources).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {"bag_path", "source", "metadata_path"}.issubset(reader.fieldnames or ()):
            raise ValueError("Invalid source CSV columns")
        rows = list(reader)
    sources = {}
    for row in rows:
        path = Path(row["bag_path"]).expanduser().resolve(strict=True)
        if path in sources:
            raise ValueError(f"Duplicate source: {path}")
        sources[path] = row
    # RichHandler may wrap a long path across lines and indent continuations.
    # Match the exact source paths after removing display whitespace, rather
    # than guessing paths from an arbitrary word following "Processing".
    compact = "".join(re.sub(r"\x1b\[[0-9;]*m", "", log).split())
    events = []
    for path, row in sources.items():
        raw_path = row["bag_path"]
        if any(c.isspace() for c in raw_path):
            raise ValueError("Source paths containing whitespace are unsupported")
        token = "Processing" + raw_path
        start = 0
        while (position := compact.find(token, start)) != -1:
            events.append((position, "start", path, 0))
            start = position + len(token)
    for match in re.finditer(r"Episode(\d+)completed:(\d+)frames", compact):
        events.append((match.start(), "done", int(match.group(1)), int(match.group(2))))
    processed = []
    pending = None
    for _, kind, value, frames in sorted(events):
        if kind == "start":
            if pending is not None:
                raise ValueError("Unfinished bag in converter log")
            pending = value
        else:
            if pending is None or value != len(processed):
                raise ValueError("Conversion completion events are incomplete or out of order")
            processed.append((pending, frames))
            pending = None
    if pending is not None or not processed or "Normalconversioncompleted!" not in compact:
        raise ValueError("Conversion log is incomplete")
    if len(processed) != len(sources) or {bag for bag, _ in processed} != set(sources):
        raise ValueError("Converter bags and source CSV do not match exactly")
    root = Path(args.dataset_root).expanduser().resolve()
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    if int(info["total_episodes"]) != len(processed):
        raise ValueError("Converted episode count differs from log")
    import pyarrow.parquet as pq
    lengths = {}
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "frame_index"])
        for episode, frame in zip(table["episode_index"].to_pylist(),
                                  table["frame_index"].to_pylist(), strict=True):
            episode, frame = int(episode), int(frame)
            lengths[episode] = max(lengths.get(episode, 0), frame + 1)
    if lengths != {i: frames for i, (_, frames) in enumerate(processed)}:
        raise ValueError("Per-episode dataset lengths do not match converter log")
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["episode_index", "source", "bag_path", "metadata_path"])
        writer.writeheader()
        for index, (bag, _) in enumerate(processed):
            writer.writerow({"episode_index": index, **{k: sources[bag][k] for k in
                            ("source", "bag_path", "metadata_path")}})
    print(f"CONVERSION_MAPPING_READY: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
