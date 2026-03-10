#!/usr/bin/env python3
"""Run AQ1 autoresearch experiments autonomously for a fixed time budget."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
REPO_PYTHON = str(REPO_ROOT / ".venv/bin/python")

from aq1_autoresearch.score_run import collect_run_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous AQ1 experiment runner")
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--baseline-run-dir", required=True)
    parser.add_argument("--results-tsv", default="aq1_autoresearch/results.tsv")
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--budget-growth-pct", type=float, default=5.0)
    parser.add_argument("--full-ppl-chunks", type=int, default=20)
    parser.add_argument("--quick-screen-margin", type=float, default=1.0)
    parser.add_argument("--max-experiments", type=int, default=64)
    parser.add_argument("--retain-checkpoints", choices=["none", "keep"], default="none")
    parser.add_argument("--checkpoint-commits", action="store_true")
    parser.add_argument("--checkpoint-dir", default="aq1_autoresearch/checkpoints")
    return parser.parse_args()


def run_cmd(cmd: list[str], log_path: Path) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"$ {' '.join(cmd)}\n")
        log.flush()
        result = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\n[exit {result.returncode}]\n")
        return result.returncode, time.time() - started_at


def log_event(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{stamp} {message}\n")


def git_commit_id() -> str:
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(["git", "diff", "--quiet"], cwd=REPO_ROOT).returncode != 0
    return f"{head}-dirty" if dirty else head


def load_existing_metrics(results_tsv: Path) -> tuple[float, float, str | None]:
    best_full = float("inf")
    baseline_quick = float("inf")
    best_run: str | None = None
    if not results_tsv.exists():
        return best_full, baseline_quick, best_run
    with open(results_tsv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            full = (row.get("full_perplexity") or "").strip()
            quick = (row.get("quick_perplexity") or "").strip()
            status = (row.get("status") or "").strip()
            change_family = (row.get("change_family") or "").strip()
            if full and status == "keep":
                full_value = float(full)
                if full_value < best_full:
                    best_full = full_value
                    best_run = (row.get("run_dir") or "").strip() or None
            if quick and change_family == "baseline":
                baseline_quick = min(baseline_quick, float(quick))
    return best_full, baseline_quick, best_run


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def append_results_row(
    results_tsv: Path,
    commit: str,
    summary: dict[str, Any],
    snap_ok: str,
    status: str,
    retention: str,
    change_family: str,
    description: str,
    timing: dict[str, Any] | None = None,
) -> None:
    timing = timing or {}
    row = [
        fmt(commit),
        fmt(summary.get("run_dir")),
        fmt(summary.get("artifact_type")),
        fmt(summary.get("candidate_checkpoint")),
        fmt(summary.get("score")),
        fmt(summary.get("full_perplexity")),
        fmt(summary.get("quick_perplexity")),
        fmt(summary.get("projected_payload_mib")),
        fmt(summary.get("avg_bits_per_weight")),
        fmt(summary.get("size_ok")),
        fmt(summary.get("best_eval_loss")),
        fmt(summary.get("final_eval_loss")),
        fmt(summary.get("elapsed_sec")),
        snap_ok,
        "n/a",
        status,
        retention,
        change_family,
        description,
        fmt(timing.get("experiment_wall_sec")),
        fmt(timing.get("proxy_wall_sec")),
        fmt(timing.get("proxy_wall_pct")),
        fmt(timing.get("full_ppl_sec")),
        fmt(timing.get("full_ppl_pct")),
        fmt(timing.get("snap_sec")),
        fmt(timing.get("snap_pct")),
        fmt(timing.get("cleanup_sec")),
        fmt(timing.get("cleanup_pct")),
    ]
    with open(results_tsv, "a", encoding="utf-8") as f:
        f.write("\t".join(row) + "\n")


def experiment_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    def add_mixedbit(
        *,
        family: str,
        scope: str,
        selection: str,
        group_size: int,
        budget_growth_pct: float,
        upgrade_bits: int | None = None,
        max_upgrade_bits: int | None = None,
    ) -> None:
        budget_tag = str(budget_growth_pct).replace(".", "p")
        if family == "mixedbit":
            assert upgrade_bits is not None
            name = f"mix_{scope}_{selection[:3]}_b{upgrade_bits}_g{group_size}_p{budget_tag}"
            description = (
                f"{scope} {upgrade_bits}-bit upgrades ranked by "
                f"{'absolute local MAE gain' if selection == 'absolute' else 'local improvement per extra bit'} "
                f"at group_size={group_size} within a {budget_growth_pct:.2f}% payload budget"
            )
        else:
            assert max_upgrade_bits is not None
            name = f"tier_{scope}_{selection[:3]}_b{max_upgrade_bits}_g{group_size}_p{budget_tag}"
            description = (
                f"{scope} tiered 4->{max_upgrade_bits}-bit upgrades via greedy "
                f"{'absolute local MAE gain' if selection == 'absolute' else 'efficiency'} "
                f"at group_size={group_size} within a {budget_growth_pct:.2f}% payload budget"
            )

        spec: dict[str, Any] = {
            "family": family,
            "name": name,
            "group_size": group_size,
            "scope": scope,
            "selection": selection,
            "budget_growth_pct": budget_growth_pct,
            "change_family": "mixedbit",
            "description": description,
        }
        if upgrade_bits is not None:
            spec["upgrade_bits"] = upgrade_bits
        if max_upgrade_bits is not None:
            spec["max_upgrade_bits"] = max_upgrade_bits
        specs.append(spec)

    priority_specs = [
        ("mixedbit_tiered", "all", "efficiency", 16, 5.0, None, 6),
        ("mixedbit_tiered", "attn", "efficiency", 16, 5.0, None, 6),
        ("mixedbit_tiered", "all", "efficiency", 16, 3.5, None, 6),
        ("mixedbit", "all", "efficiency", 16, 3.5, 5, None),
        ("mixedbit", "all", "absolute", 16, 5.0, 5, None),
        ("mixedbit_tiered", "all", "absolute", 16, 5.0, None, 6),
        ("mixedbit", "attn", "efficiency", 16, 3.5, 5, None),
        ("mixedbit_tiered", "mlp", "efficiency", 16, 5.0, None, 6),
        ("mixedbit_tiered", "all", "efficiency", 8, 5.0, None, 6),
        ("mixedbit", "all", "efficiency", 8, 5.0, 5, None),
    ]
    for family, scope, selection, group_size, budget_growth_pct, upgrade_bits, max_upgrade_bits in priority_specs:
        add_mixedbit(
            family=family,
            scope=scope,
            selection=selection,
            group_size=group_size,
            budget_growth_pct=budget_growth_pct,
            upgrade_bits=upgrade_bits,
            max_upgrade_bits=max_upgrade_bits,
        )

    for group_size in (16, 8, 32):
        for budget_growth_pct in (2.5, 3.5, 5.0):
            for scope in ("all", "attn", "mlp"):
                for selection in ("efficiency", "absolute"):
                    add_mixedbit(
                        family="mixedbit",
                        scope=scope,
                        selection=selection,
                        group_size=group_size,
                        budget_growth_pct=budget_growth_pct,
                        upgrade_bits=5,
                    )

    for group_size in (16, 8, 32):
        for budget_growth_pct in (2.5, 3.5, 5.0):
            for scope in ("all", "attn", "mlp"):
                for selection in ("efficiency", "absolute"):
                    add_mixedbit(
                        family="mixedbit_tiered",
                        scope=scope,
                        selection=selection,
                        group_size=group_size,
                        budget_growth_pct=budget_growth_pct,
                        max_upgrade_bits=6,
                    )

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in specs:
        if spec["name"] in seen:
            continue
        deduped.append(spec)
        seen.add(spec["name"])
    return deduped


def relative_repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def commit_checkpoint(
    *,
    checkpoint_dir: Path,
    output_root: Path,
    results_tsv: Path,
    run_dir: Path,
    spec: dict[str, Any],
    summary: dict[str, Any],
    status: str,
    retention: str,
    runner_log: Path,
    final: bool = False,
) -> None:
    campaign_dir = checkpoint_dir / output_root.name
    record = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "run_dir": str(run_dir),
        "status": status,
        "retention": retention,
        "final": final,
        "spec": spec,
        "summary": summary,
    }
    if final:
        record["best_run"] = summary.get("best_run")
        snapshot_path = campaign_dir / "final.json"
    else:
        snapshot_path = campaign_dir / f"{run_dir.name}.json"
    latest_path = campaign_dir / "latest.json"
    write_json(snapshot_path, record)
    write_json(latest_path, record)

    paths = [relative_repo_path(results_tsv), relative_repo_path(snapshot_path), relative_repo_path(latest_path)]
    subprocess.run(["git", "add", "--", *paths], cwd=REPO_ROOT, check=True)
    staged = subprocess.run(["git", "diff", "--cached", "--quiet", "--", *paths], cwd=REPO_ROOT).returncode != 0
    if not staged:
        return

    metric = summary.get("full_perplexity")
    if metric is None:
        metric = summary.get("quick_perplexity")
    if metric is None:
        metric = summary.get("score")
    suffix = f" p={fmt(metric)}" if metric is not None else ""
    subject = f"AQ1 checkpoint: {output_root.name} final" if final else f"AQ1 checkpoint: {run_dir.name}{suffix}"
    result = subprocess.run(
        ["git", "commit", "-m", subject, "--", *paths],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        log_event(runner_log, f"checkpoint commit created: {subject}")
    else:
        log_event(runner_log, f"checkpoint commit failed: {subject}")
        if result.stdout.strip():
            log_event(runner_log, result.stdout.strip())
        if result.stderr.strip():
            log_event(runner_log, result.stderr.strip())


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results_tsv = Path(args.results_tsv).expanduser().resolve()
    baseline_run_dir = Path(args.baseline_run_dir).expanduser().resolve()
    baseline_checkpoint = baseline_run_dir / "v2_tightened.pt"
    baseline_quick_log = collect_run_summary(run_dir=baseline_run_dir)
    best_full, baseline_quick, best_run = load_existing_metrics(results_tsv)
    if baseline_quick == float("inf"):
        baseline_quick = baseline_quick_log.get("quick_perplexity") or float("inf")

    runner_state = output_root / "runner_state.json"
    runner_log = output_root / "runner.log"
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    started_at = time.time()
    commit = git_commit_id()
    specs = experiment_specs()[: args.max_experiments]
    completed_experiments = 0
    log_event(
        runner_log,
        f"starting run commit={commit} specs={len(specs)} hours={args.hours} retain={args.retain_checkpoints}",
    )

    for idx, spec in enumerate(specs, start=1):
        elapsed = time.time() - started_at
        if elapsed >= args.hours * 3600:
            log_event(runner_log, f"stopping: reached time budget after {completed_experiments} experiments")
            break

        run_dir = output_root / f"exp_{idx:03d}_{spec['name']}"
        if run_dir.exists():
            run_dir = output_root / f"exp_{idx:03d}_{spec['name']}_{int(time.time())}"
        run_dir.mkdir(parents=True, exist_ok=True)
        completed_experiments += 1

        cache_tag = (
            f"tiered_b{spec.get('max_upgrade_bits', 6)}"
            if spec["family"] == "mixedbit_tiered"
            else f"b{spec.get('upgrade_bits', 5)}"
        )
        cache_json = output_root / f"mixedbit_cache_g{spec['group_size']}_{spec.get('scope', 'all')}_{cache_tag}.json"
        spec_budget = float(spec.get("budget_growth_pct", args.budget_growth_pct))
        build_cmd = [
            REPO_PYTHON,
            "aq1_autoresearch/build_candidate.py",
            "--family",
            spec["family"],
            "--output",
            str(run_dir),
            "--model-id",
            args.model_id,
            "--group-size",
            str(spec["group_size"]),
        ]
        if spec["family"] in {"mixedbit", "mixedbit_tiered"}:
            build_cmd.extend(
                [
                    "--baseline-checkpoint",
                    str(baseline_checkpoint),
                    "--budget-growth-pct",
                    str(spec_budget),
                    "--scope",
                    spec["scope"],
                    "--selection",
                    spec["selection"],
                    "--cache-json",
                    str(cache_json),
                ]
            )
            if spec["family"] == "mixedbit":
                build_cmd.extend(["--upgrade-bits", str(spec["upgrade_bits"])])
            else:
                build_cmd.extend(["--max-upgrade-bits", str(spec["max_upgrade_bits"])])
        else:
            build_cmd.extend(["--permute-strategy", spec["permute_strategy"]])

        init_log = run_dir / "init.log"
        status = "crash"
        snap_ok = "skipped"
        retention = "discard"
        timing = {
            "experiment_wall_sec": None,
            "proxy_wall_sec": None,
            "proxy_wall_pct": None,
            "full_ppl_sec": None,
            "full_ppl_pct": None,
            "snap_sec": None,
            "snap_pct": None,
            "cleanup_sec": None,
            "cleanup_pct": None,
        }
        experiment_started_at = time.time()
        log_event(runner_log, f"experiment {idx:03d} starting {run_dir.name}: {spec['description']}")

        build_exit, build_sec = run_cmd(build_cmd, init_log)
        timing["proxy_wall_sec"] = build_sec
        log_event(runner_log, f"experiment {idx:03d} build exit={build_exit} sec={fmt(build_sec)}")
        if build_exit == 0:
            quick_summary = collect_run_summary(run_dir=run_dir)
            quick = quick_summary.get("quick_perplexity")
            screen_fail = baseline_quick != float("inf") and quick is not None and quick > baseline_quick + args.quick_screen_margin

            if not screen_fail:
                ppl_cmd = [
                    REPO_PYTHON,
                    "scripts/measure_perplexity.py",
                    str(run_dir / "v2_tightened.pt"),
                    "--config",
                    str(run_dir / "config.json"),
                    "--device",
                    "mps" if sys.platform == "darwin" else "cpu",
                    "--dtype",
                    "fp16",
                    "--max-chunks",
                    str(args.full_ppl_chunks),
                ]
                ppl_exit, ppl_sec = run_cmd(ppl_cmd, run_dir / "perplexity.log")
                timing["full_ppl_sec"] = ppl_sec
                log_event(runner_log, f"experiment {idx:03d} full-ppl exit={ppl_exit} sec={fmt(ppl_sec)}")

                snap_cmd = [
                    REPO_PYTHON,
                    "scripts/snap_and_test_v2.py",
                    "--checkpoint",
                    str(run_dir / "v2_tightened.pt"),
                    "--fp16",
                    "--no-test",
                    "--output",
                    str(run_dir / "snapped_fp16.pt"),
                ]
                snap_exit, snap_sec = run_cmd(snap_cmd, run_dir / "snap.log")
                timing["snap_sec"] = snap_sec
                log_event(runner_log, f"experiment {idx:03d} snap exit={snap_exit} sec={fmt(snap_sec)}")
                snap_ok = "true" if (run_dir / "snapped_fp16.pt").exists() else "false"
            else:
                snap_ok = "skipped"
                log_event(
                    runner_log,
                    f"experiment {idx:03d} screened out quick_ppl={fmt(quick)} baseline_quick={fmt(baseline_quick)}",
                )

            summary = collect_run_summary(
                run_dir=run_dir,
                baseline_run_dir=str(baseline_run_dir),
                max_size_growth_pct=spec_budget,
            )

            if screen_fail:
                status = "discard"
                retention = "discard"
            else:
                full_ppl = summary.get("full_perplexity")
                size_ok = summary.get("size_ok")
                is_keep = (
                    full_ppl is not None
                    and snap_ok == "true"
                    and size_ok is not False
                    and full_ppl < best_full
                )
                status = "keep" if is_keep else "discard"
                retention = "keep" if is_keep and args.retain_checkpoints == "keep" else "discard"
                if is_keep:
                    best_full = float(full_ppl)
                    best_run = str(run_dir)

            log_event(
                runner_log,
                f"experiment {idx:03d} result status={status} full={fmt(summary.get('full_perplexity'))} "
                f"quick={fmt(summary.get('quick_perplexity'))} size_ok={fmt(summary.get('size_ok'))}",
            )

            cleanup_cmd = [
                REPO_PYTHON,
                "aq1_autoresearch/cleanup_run.py",
                "--run-dir",
                str(run_dir),
                "--status",
                "keep" if retention == "keep" else "discard",
            ]
            if args.retain_checkpoints == "none" or retention != "keep":
                cleanup_cmd.extend(["--retain-checkpoint", "none"])
            cleanup_exit, cleanup_sec = run_cmd(cleanup_cmd, run_dir / "cleanup.log")
            timing["cleanup_sec"] = cleanup_sec
            log_event(
                runner_log,
                f"experiment {idx:03d} cleanup exit={cleanup_exit} retention={retention} sec={fmt(cleanup_sec)}",
            )
        else:
            summary = collect_run_summary(run_dir=run_dir)
            log_event(runner_log, f"experiment {idx:03d} crashed during build")

        timing["experiment_wall_sec"] = time.time() - experiment_started_at
        total_wall = timing["experiment_wall_sec"] or 0.0
        for sec_key, pct_key in (
            ("proxy_wall_sec", "proxy_wall_pct"),
            ("full_ppl_sec", "full_ppl_pct"),
            ("snap_sec", "snap_pct"),
            ("cleanup_sec", "cleanup_pct"),
        ):
            sec_value = timing.get(sec_key)
            if sec_value is not None and total_wall > 0:
                timing[pct_key] = 100.0 * float(sec_value) / total_wall

        append_results_row(
            results_tsv=results_tsv,
            commit=commit,
            summary=summary,
            snap_ok=snap_ok,
            status=status,
            retention=retention,
            change_family=spec["change_family"],
            description=spec["description"],
            timing=timing,
        )

        if args.checkpoint_commits and status == "keep":
            commit_checkpoint(
                checkpoint_dir=checkpoint_dir,
                output_root=output_root,
                results_tsv=results_tsv,
                run_dir=run_dir,
                spec=spec,
                summary=summary,
                status=status,
                retention=retention,
                runner_log=runner_log,
            )

        log_event(
            runner_log,
            f"experiment {idx:03d} wall_sec={fmt(timing['experiment_wall_sec'])} "
            f"proxy={fmt(timing['proxy_wall_sec'])}/{fmt(timing['proxy_wall_pct'])}% "
            f"full={fmt(timing['full_ppl_sec'])}/{fmt(timing['full_ppl_pct'])}% "
            f"snap={fmt(timing['snap_sec'])}/{fmt(timing['snap_pct'])}% "
            f"cleanup={fmt(timing['cleanup_sec'])}/{fmt(timing['cleanup_pct'])}%",
        )

        write_state = {
            "last_run": str(run_dir),
            "last_status": status,
            "best_full_perplexity": None if best_full == float("inf") else best_full,
            "best_run": best_run,
            "completed_experiments": completed_experiments,
            "elapsed_hours": (time.time() - started_at) / 3600.0,
        }
        with open(runner_state, "w", encoding="utf-8") as f:
            json.dump(write_state, f, indent=2, sort_keys=True)

    if args.checkpoint_commits:
        final_summary = {
            "best_full_perplexity": None if best_full == float("inf") else best_full,
            "best_run": best_run,
            "completed_experiments": completed_experiments,
            "elapsed_hours": (time.time() - started_at) / 3600.0,
        }
        commit_checkpoint(
            checkpoint_dir=checkpoint_dir,
            output_root=output_root,
            results_tsv=results_tsv,
            run_dir=output_root,
            spec={"kind": "final"},
            summary=final_summary,
            status="final",
            retention=args.retain_checkpoints,
            runner_log=runner_log,
            final=True,
        )
    log_event(
        runner_log,
        f"finished run experiments={completed_experiments} best_full={fmt(None if best_full == float('inf') else best_full)}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
