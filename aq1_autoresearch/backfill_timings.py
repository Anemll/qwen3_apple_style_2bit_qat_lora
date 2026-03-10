#!/usr/bin/env python3
"""Backfill AQ1 experiment timing columns from runner logs."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


TIMING_FIELDS = [
    "experiment_wall_sec",
    "proxy_wall_sec",
    "proxy_wall_pct",
    "full_ppl_sec",
    "full_ppl_pct",
    "snap_sec",
    "snap_pct",
    "cleanup_sec",
    "cleanup_pct",
]

BASE_FIELDS = [
    "commit",
    "run_dir",
    "artifact_type",
    "candidate_checkpoint",
    "score",
    "full_perplexity",
    "quick_perplexity",
    "projected_payload_mib",
    "avg_bits_per_weight",
    "size_ok",
    "best_eval_loss",
    "final_eval_loss",
    "elapsed_sec",
    "snap_ok",
    "inference_ok",
    "status",
    "retention",
    "change_family",
    "description",
]

HEADER = BASE_FIELDS + TIMING_FIELDS

EVENT_PATTERNS = {
    "start": re.compile(r"^(?P<ts>\S+) experiment \d+ starting (?P<name>\S+): "),
    "build": re.compile(r"^(?P<ts>\S+) experiment \d+ build exit="),
    "full": re.compile(r"^(?P<ts>\S+) experiment \d+ full-ppl exit="),
    "snap": re.compile(r"^(?P<ts>\S+) experiment \d+ snap exit="),
    "result": re.compile(r"^(?P<ts>\S+) experiment \d+ result status="),
    "cleanup": re.compile(r"^(?P<ts>\S+) experiment \d+ cleanup exit="),
    "wall": re.compile(
        r"^(?P<ts>\S+) experiment \d+ wall_sec=(?P<wall>[0-9.]+) "
        r"proxy=(?P<proxy_sec>[0-9.]+)/(?P<proxy_pct>[0-9.]+)% "
        r"full=(?P<full_sec>[0-9.]+)/(?P<full_pct>[0-9.]+)% "
        r"snap=(?P<snap_sec>[0-9.]+)/(?P<snap_pct>[0-9.]+)% "
        r"cleanup=(?P<cleanup_sec>[0-9.]+)/(?P<cleanup_pct>[0-9.]+)%"
    ),
}


@dataclass
class ExperimentEvents:
    name: str
    start: datetime | None = None
    build: datetime | None = None
    full: datetime | None = None
    snap: datetime | None = None
    result: datetime | None = None
    cleanup: datetime | None = None
    explicit: dict[str, float] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill timing columns in aq1_autoresearch/results.tsv")
    parser.add_argument("--results-tsv", default="aq1_autoresearch/results.tsv")
    parser.add_argument("--write", action="store_true", help="Rewrite the TSV in place")
    return parser.parse_args()


def parse_ts(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def seconds(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    return max(0.0, (end - start).total_seconds())


def fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}".rstrip("0").rstrip(".")


def load_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(row) for row in reader]


def runner_log_for_run(run_dir: Path) -> Path | None:
    run_dir = run_dir.expanduser().resolve()
    for parent in [run_dir.parent, *run_dir.parents]:
        candidate = parent / "runner.log"
        if candidate.exists():
            return candidate
    return None


def parse_runner_log(path: Path) -> dict[str, ExperimentEvents]:
    by_name: dict[str, ExperimentEvents] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        current_name: str | None = None
        for line in f:
            line = line.rstrip("\n")
            start_match = EVENT_PATTERNS["start"].match(line)
            if start_match:
                current_name = start_match.group("name")
                event = by_name.setdefault(current_name, ExperimentEvents(name=current_name))
                event.start = parse_ts(start_match.group("ts"))
                continue

            wall_match = EVENT_PATTERNS["wall"].match(line)
            if wall_match and current_name:
                event = by_name.setdefault(current_name, ExperimentEvents(name=current_name))
                event.explicit = {
                    "experiment_wall_sec": float(wall_match.group("wall")),
                    "proxy_wall_sec": float(wall_match.group("proxy_sec")),
                    "proxy_wall_pct": float(wall_match.group("proxy_pct")),
                    "full_ppl_sec": float(wall_match.group("full_sec")),
                    "full_ppl_pct": float(wall_match.group("full_pct")),
                    "snap_sec": float(wall_match.group("snap_sec")),
                    "snap_pct": float(wall_match.group("snap_pct")),
                    "cleanup_sec": float(wall_match.group("cleanup_sec")),
                    "cleanup_pct": float(wall_match.group("cleanup_pct")),
                }
                continue

            for key in ("build", "full", "snap", "result", "cleanup"):
                match = EVENT_PATTERNS[key].match(line)
                if match and current_name:
                    event = by_name.setdefault(current_name, ExperimentEvents(name=current_name))
                    setattr(event, key, parse_ts(match.group("ts")))
                    break
    return by_name


def derive_timings(events: ExperimentEvents) -> dict[str, str]:
    if events.explicit:
        return {field: fmt_float(events.explicit.get(field)) for field in TIMING_FIELDS}

    experiment_wall_sec = seconds(events.start, events.cleanup or events.result or events.snap or events.full or events.build)
    proxy_wall_sec = seconds(events.start, events.build)
    full_ppl_sec = seconds(events.build, events.full)
    snap_sec = seconds(events.full, events.snap)
    cleanup_sec = seconds(events.result or events.snap or events.full or events.build, events.cleanup)

    values = {
        "experiment_wall_sec": experiment_wall_sec,
        "proxy_wall_sec": proxy_wall_sec,
        "proxy_wall_pct": None,
        "full_ppl_sec": full_ppl_sec,
        "full_ppl_pct": None,
        "snap_sec": snap_sec,
        "snap_pct": None,
        "cleanup_sec": cleanup_sec,
        "cleanup_pct": None,
    }
    if experiment_wall_sec and experiment_wall_sec > 0:
        for sec_key, pct_key in (
            ("proxy_wall_sec", "proxy_wall_pct"),
            ("full_ppl_sec", "full_ppl_pct"),
            ("snap_sec", "snap_pct"),
            ("cleanup_sec", "cleanup_pct"),
        ):
            sec_value = values.get(sec_key)
            if sec_value is not None:
                values[pct_key] = 100.0 * float(sec_value) / experiment_wall_sec
    return {field: fmt_float(values.get(field)) for field in TIMING_FIELDS}


def enrich_rows(rows: list[dict[str, str]], repo_root: Path) -> tuple[list[dict[str, str]], int]:
    cache: dict[Path, dict[str, ExperimentEvents]] = {}
    updated = 0

    for row in rows:
        run_dir_value = (row.get("run_dir") or "").strip()
        if not run_dir_value:
            continue
        run_dir = Path(run_dir_value)
        if not run_dir.is_absolute():
            run_dir = (repo_root / run_dir).resolve()
        runner_log = runner_log_for_run(run_dir)
        if runner_log is None:
            for field in TIMING_FIELDS:
                row.setdefault(field, "")
            continue
        events_by_name = cache.setdefault(runner_log, parse_runner_log(runner_log))
        events = events_by_name.get(run_dir.name)
        if events is None:
            for field in TIMING_FIELDS:
                row.setdefault(field, "")
            continue
        derived = derive_timings(events)
        changed = False
        for field, value in derived.items():
            if row.get(field, "") != value:
                row[field] = value
                changed = True
        if changed:
            updated += 1
    return rows, updated


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in HEADER})


def main() -> int:
    args = parse_args()
    results_tsv = Path(args.results_tsv).expanduser().resolve()
    repo_root = results_tsv.parent.parent.resolve()
    rows = load_rows(results_tsv)
    rows, updated = enrich_rows(rows, repo_root)

    if args.write:
        write_rows(results_tsv, rows)

    print(f"rows={len(rows)} updated={updated} write={'true' if args.write else 'false'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
