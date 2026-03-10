#!/usr/bin/env python3
"""Check AQ1 autoresearch live status, report freshness, and current summary."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_TSV = REPO_ROOT / "aq1_autoresearch" / "results.tsv"
REPORT_MD = REPO_ROOT / "aq1_autoresearch" / "progress_report.md"
REPORT_SVG = REPO_ROOT / "aq1_autoresearch" / "progress_report.svg"
REPORT_SCRIPT = REPO_ROOT / "aq1_autoresearch" / "render_progress_report.py"
PERPLEXITY_JSON = REPO_ROOT / "results" / "perplexity.json"
RUNS_ROOT = REPO_ROOT / "runs" / "aq1_auto" / "qwen06b-init-ppl"
RUNNER_MARKERS = (
    "aq1_autoresearch/auto_runner.py",
    "aq1_autoresearch/build_candidate.py",
    "scripts/measure_perplexity.py",
    "scripts/snap_and_test_v2.py",
)


@dataclass
class RunningProcess:
    pid: int
    ppid: int
    state: str
    etime: str
    command: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check AQ1 autoresearch status")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    return parser.parse_args()


def maybe_float(value: str | None) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    return float(value)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def format_ts(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def find_running_processes() -> list[RunningProcess]:
    result = subprocess.run(
        ["ps", "ax", "-o", "pid=,ppid=,state=,etime=,command="],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    processes: list[RunningProcess] = []
    for line in result.stdout.splitlines():
        raw = line.strip()
        if not raw:
            continue
        parts = raw.split(None, 4)
        if len(parts) != 5:
            continue
        pid_s, ppid_s, state, etime, command = parts
        if not any(marker in command for marker in RUNNER_MARKERS):
            continue
        try:
            if "check_status.py" in command:
                continue
            processes.append(
                RunningProcess(
                    pid=int(pid_s),
                    ppid=int(ppid_s),
                    state=state,
                    etime=etime,
                    command=command,
                )
            )
        except ValueError:
            continue
    return sorted(processes, key=lambda proc: (proc.ppid, proc.pid))


def detect_active_run(processes: list[RunningProcess]) -> Path | None:
    auto_runner = next((proc for proc in processes if "aq1_autoresearch/auto_runner.py" in proc.command), None)
    if auto_runner:
        marker = "--output-root "
        if marker in auto_runner.command:
            tail = auto_runner.command.split(marker, 1)[1]
            output_root = tail.split(" --", 1)[0].strip()
            path = Path(output_root)
            if not path.is_absolute():
                path = (REPO_ROOT / output_root).resolve()
            return path

    runner_states = sorted(RUNS_ROOT.glob("*/runner_state.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if runner_states:
        return runner_states[0].parent
    return None


def collect_results_summary(results_tsv: Path) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    with open(results_tsv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)

    baseline = None
    best_row = None
    latest_row = rows[-1] if rows else None
    full_eval_count = 0
    quick_only_count = 0

    for row in rows:
        full = maybe_float(row.get("full_perplexity"))
        quick = maybe_float(row.get("quick_perplexity"))
        if row.get("change_family") == "baseline" and full is not None and baseline is None:
            baseline = full
        if full is not None:
            full_eval_count += 1
            if best_row is None or full < maybe_float(best_row.get("full_perplexity")):
                best_row = row
        elif quick is not None:
            quick_only_count += 1

    if baseline is None and rows:
        baseline = maybe_float(rows[0].get("full_perplexity"))

    best_full = maybe_float(best_row.get("full_perplexity")) if best_row else None
    improvement = baseline - best_full if baseline is not None and best_full is not None else None

    return {
        "row_count": len(rows),
        "full_eval_count": full_eval_count,
        "quick_only_count": quick_only_count,
        "baseline_full_perplexity": baseline,
        "best_full_perplexity": best_full,
        "best_run": best_row.get("run_dir") if best_row else None,
        "best_description": best_row.get("description") if best_row else None,
        "best_avg_bits_per_weight": maybe_float(best_row.get("avg_bits_per_weight")) if best_row else None,
        "best_projected_payload_mib": maybe_float(best_row.get("projected_payload_mib")) if best_row else None,
        "improvement_vs_baseline": improvement,
        "latest_run": latest_row.get("run_dir") if latest_row else None,
        "latest_status": latest_row.get("status") if latest_row else None,
        "latest_score": maybe_float(latest_row.get("score")) if latest_row else None,
        "latest_description": latest_row.get("description") if latest_row else None,
    }


def load_original_reference() -> dict[str, Any] | None:
    data = load_json(PERPLEXITY_JSON)
    if not data:
        return None
    entry = data.get("baseline:Qwen/Qwen3-0.6B")
    if not isinstance(entry, dict) or entry.get("perplexity") is None:
        return None
    return {
        "perplexity": float(entry["perplexity"]),
        "cross_entropy": entry.get("cross_entropy"),
        "tokens": entry.get("tokens"),
        "time_seconds": entry.get("time_seconds"),
        "dtype": entry.get("dtype"),
    }


def freshness(target: Path, dependencies: list[Path]) -> dict[str, Any]:
    existing_deps = [dep for dep in dependencies if dep.exists()]
    latest_dep = max(existing_deps, key=lambda path: path.stat().st_mtime) if existing_deps else None
    target_exists = target.exists()
    target_mtime = target.stat().st_mtime if target_exists else None
    latest_dep_mtime = latest_dep.stat().st_mtime if latest_dep else None
    fresh = bool(target_exists and (latest_dep_mtime is None or target_mtime >= latest_dep_mtime))
    lag_seconds = max(0.0, (latest_dep_mtime - target_mtime)) if target_exists and latest_dep_mtime is not None else None
    return {
        "path": str(target),
        "exists": target_exists,
        "fresh": fresh,
        "target_updated_at": format_ts(target),
        "latest_dependency": str(latest_dep) if latest_dep else None,
        "latest_dependency_updated_at": format_ts(latest_dep),
        "lag_seconds": lag_seconds,
    }


def build_status() -> dict[str, Any]:
    processes = find_running_processes()
    active_run = detect_active_run(processes)
    runner_state = load_json(active_run / "runner_state.json") if active_run else None
    runner_log = active_run / "runner.log" if active_run else None

    results_summary = collect_results_summary(RESULTS_TSV)
    original_ref = load_original_reference()
    if original_ref and results_summary["best_full_perplexity"] is not None:
        results_summary["best_delta_vs_original"] = results_summary["best_full_perplexity"] - original_ref["perplexity"]
        if results_summary["baseline_full_perplexity"] is not None:
            results_summary["baseline_delta_vs_original"] = results_summary["baseline_full_perplexity"] - original_ref["perplexity"]

    report_deps = [RESULTS_TSV, REPORT_SCRIPT, PERPLEXITY_JSON]
    if active_run:
        report_deps.append(active_run / "runner_state.json")

    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "running": {
            "any": bool(processes),
            "count": len(processes),
            "active_run": str(active_run) if active_run else None,
            "processes": [
                {
                    "pid": proc.pid,
                    "ppid": proc.ppid,
                    "state": proc.state,
                    "elapsed": proc.etime,
                    "command": proc.command,
                }
                for proc in processes
            ],
        },
        "report_freshness": {
            "markdown": freshness(REPORT_MD, report_deps),
            "svg": freshness(REPORT_SVG, report_deps),
        },
        "summary": results_summary,
        "runner_state": runner_state,
        "original_model": original_ref,
        "paths": {
            "results_tsv": str(RESULTS_TSV),
            "report_md": str(REPORT_MD),
            "report_svg": str(REPORT_SVG),
            "perplexity_json": str(PERPLEXITY_JSON),
            "runner_log": str(runner_log) if runner_log else None,
        },
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def print_text(status: dict[str, Any]) -> None:
    running = status["running"]
    summary = status["summary"]
    runner_state = status.get("runner_state") or {}
    original = status.get("original_model") or {}

    print("AQ1 ANE-Native Quantization Status")
    print(f"checked_at:            {status['checked_at']}")
    print()
    print("Running")
    print(f"  any:                 {running['any']}")
    print(f"  count:               {running['count']}")
    print(f"  active_run:          {running['active_run'] or '-'}")
    for proc in running["processes"]:
        print(f"  pid {proc['pid']:>5} [{proc['state']}] {proc['elapsed']:>8}  {proc['command']}")
    print()
    print("Report Freshness")
    for label in ("markdown", "svg"):
        item = status["report_freshness"][label]
        print(f"  {label}:")
        print(f"    fresh:             {item['fresh']}")
        print(f"    updated_at:        {item['target_updated_at'] or '-'}")
        print(f"    latest_dep:        {item['latest_dependency'] or '-'}")
        print(f"    dep_updated_at:    {item['latest_dependency_updated_at'] or '-'}")
        print(f"    lag_seconds:       {fmt(item['lag_seconds'])}")
    print()
    print("Summary")
    print(f"  rows_logged:         {summary['row_count']}")
    print(f"  full_evals:          {summary['full_eval_count']}")
    print(f"  quick_only:          {summary['quick_only_count']}")
    print(f"  baseline_full_ppl:   {fmt(summary['baseline_full_perplexity'])}")
    print(f"  best_full_ppl:       {fmt(summary['best_full_perplexity'])}")
    print(f"  improvement_vs_q4:   {fmt(summary['improvement_vs_baseline'])}")
    print(f"  best_delta_orig:     {fmt(summary.get('best_delta_vs_original'))}")
    print(f"  best_run:            {summary['best_run'] or '-'}")
    print(f"  best_desc:           {summary['best_description'] or '-'}")
    print(f"  best_size_mib:       {fmt(summary['best_projected_payload_mib'])}")
    print(f"  best_avg_bits:       {fmt(summary['best_avg_bits_per_weight'])}")
    print(f"  latest_run:          {summary['latest_run'] or '-'}")
    print(f"  latest_status:       {summary['latest_status'] or '-'}")
    print(f"  latest_score:        {fmt(summary['latest_score'])}")
    print(f"  latest_desc:         {summary['latest_description'] or '-'}")
    if original:
        print(f"  original_qwen_ppl:   {fmt(original.get('perplexity'))}")
    if runner_state:
        print()
        print("Active Runner State")
        print(f"  completed:           {runner_state.get('completed_experiments', '-')}")
        print(f"  best_full:           {runner_state.get('best_full_perplexity', '-')}")
        print(f"  best_run:            {runner_state.get('best_run', '-')}")
        print(f"  last_run:            {runner_state.get('last_run', '-')}")
        print(f"  last_status:         {runner_state.get('last_status', '-')}")
        print(f"  elapsed_hours:       {fmt(runner_state.get('elapsed_hours'))}")


def main() -> int:
    args = parse_args()
    status = build_status()
    if args.format == "json":
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        print_text(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
