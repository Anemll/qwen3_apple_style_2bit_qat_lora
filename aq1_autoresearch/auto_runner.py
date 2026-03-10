#!/usr/bin/env python3
"""Run AQ1 autoresearch experiments autonomously for a fixed time budget."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

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
    parser.add_argument("--max-experiments", type=int, default=24)
    return parser.parse_args()


def run_cmd(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"$ {' '.join(cmd)}\n")
        log.flush()
        result = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\n[exit {result.returncode}]\n")
        return result.returncode


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


def load_existing_metrics(results_tsv: Path) -> tuple[float, float]:
    best_full = float("inf")
    baseline_quick = float("inf")
    if not results_tsv.exists():
        return best_full, baseline_quick
    with open(results_tsv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            full = (row.get("full_perplexity") or "").strip()
            quick = (row.get("quick_perplexity") or "").strip()
            status = (row.get("status") or "").strip()
            change_family = (row.get("change_family") or "").strip()
            if full and status == "keep":
                best_full = min(best_full, float(full))
            if quick and change_family == "baseline":
                baseline_quick = min(baseline_quick, float(quick))
    return best_full, baseline_quick


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
) -> None:
    row = [
        commit,
        summary.get("run_dir"),
        summary.get("artifact_type"),
        summary.get("candidate_checkpoint"),
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
    ]
    with open(results_tsv, "a", encoding="utf-8") as f:
        f.write("\t".join(row) + "\n")


def experiment_specs() -> list[dict[str, Any]]:
    return [
        {
            "family": "mixedbit",
            "name": "mix_attn_eff_b5_g16",
            "group_size": 16,
            "scope": "attn",
            "selection": "efficiency",
            "upgrade_bits": 5,
            "change_family": "mixedbit",
            "description": "attention-only 5-bit upgrades ranked by local improvement per extra bit",
        },
        {
            "family": "mixedbit",
            "name": "mix_all_eff_b5_g16",
            "group_size": 16,
            "scope": "all",
            "selection": "efficiency",
            "upgrade_bits": 5,
            "change_family": "mixedbit",
            "description": "global 5-bit upgrades ranked by local improvement per extra bit",
        },
        {
            "family": "mixedbit",
            "name": "mix_mlp_eff_b5_g16",
            "group_size": 16,
            "scope": "mlp",
            "selection": "efficiency",
            "upgrade_bits": 5,
            "change_family": "mixedbit",
            "description": "MLP-only 5-bit upgrades ranked by local improvement per extra bit",
        },
        {
            "family": "mixedbit",
            "name": "mix_attn_abs_b5_g16",
            "group_size": 16,
            "scope": "attn",
            "selection": "absolute",
            "upgrade_bits": 5,
            "change_family": "mixedbit",
            "description": "attention-only 5-bit upgrades ranked by absolute local MAE gain",
        },
        {
            "family": "mixedbit",
            "name": "mix_attn_eff_b6_g16",
            "group_size": 16,
            "scope": "attn",
            "selection": "efficiency",
            "upgrade_bits": 6,
            "change_family": "mixedbit",
            "description": "attention-only 6-bit upgrades within the same global size budget",
        },
        {
            "family": "mlp_permute",
            "name": "perm_combined_desc_g16",
            "group_size": 16,
            "permute_strategy": "combined_desc",
            "change_family": "folded_permute",
            "description": "folded MLP channel permutation by combined norm, descending",
        },
        {
            "family": "mlp_permute",
            "name": "perm_combined_hilo_g16",
            "group_size": 16,
            "permute_strategy": "combined_hilo",
            "change_family": "folded_permute",
            "description": "folded MLP channel permutation by combined norm, hi/lo interleave",
        },
        {
            "family": "mlp_permute",
            "name": "perm_down_desc_g16",
            "group_size": 16,
            "permute_strategy": "down_desc",
            "change_family": "folded_permute",
            "description": "folded MLP channel permutation using down_proj column norms, descending",
        },
        {
            "family": "mlp_permute",
            "name": "perm_down_hilo_g16",
            "group_size": 16,
            "permute_strategy": "down_hilo",
            "change_family": "folded_permute",
            "description": "folded MLP channel permutation using down_proj column norms, hi/lo interleave",
        },
        {
            "family": "mixedbit",
            "name": "mix_attn_eff_b5_g32",
            "group_size": 32,
            "scope": "attn",
            "selection": "efficiency",
            "upgrade_bits": 5,
            "change_family": "mixedbit",
            "description": "attention-only 5-bit upgrades with group_size=32",
        },
        {
            "family": "mlp_permute",
            "name": "perm_combined_desc_g32",
            "group_size": 32,
            "permute_strategy": "combined_desc",
            "change_family": "folded_permute",
            "description": "folded MLP channel permutation by combined norm at group_size=32",
        },
        {
            "family": "mixedbit",
            "name": "mix_all_eff_b6_g16",
            "group_size": 16,
            "scope": "all",
            "selection": "efficiency",
            "upgrade_bits": 6,
            "change_family": "mixedbit",
            "description": "global 6-bit upgrades within the same global size budget",
        },
    ]


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results_tsv = Path(args.results_tsv).expanduser().resolve()
    baseline_run_dir = Path(args.baseline_run_dir).expanduser().resolve()
    baseline_checkpoint = baseline_run_dir / "v2_tightened.pt"
    baseline_quick_log = collect_run_summary(run_dir=baseline_run_dir)
    best_full, baseline_quick = load_existing_metrics(results_tsv)
    if baseline_quick == float("inf"):
        baseline_quick = baseline_quick_log.get("quick_perplexity") or float("inf")

    runner_log = output_root / "runner_state.json"
    started_at = time.time()
    commit = git_commit_id()
    specs = experiment_specs()[: args.max_experiments]

    for idx, spec in enumerate(specs, start=1):
        elapsed = time.time() - started_at
        if elapsed >= args.hours * 3600:
            break

        run_dir = output_root / f"exp_{idx:03d}_{spec['name']}"
        if run_dir.exists():
            run_dir = output_root / f"exp_{idx:03d}_{spec['name']}_{int(time.time())}"
        run_dir.mkdir(parents=True, exist_ok=True)

        cache_json = output_root / f"mixedbit_cache_g{spec['group_size']}_{spec.get('scope', 'all')}_b{spec.get('upgrade_bits', 5)}.json"
        build_cmd = [
            sys.executable,
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
        if spec["family"] == "mixedbit":
            build_cmd.extend(
                [
                    "--baseline-checkpoint",
                    str(baseline_checkpoint),
                    "--budget-growth-pct",
                    str(args.budget_growth_pct),
                    "--scope",
                    spec["scope"],
                    "--selection",
                    spec["selection"],
                    "--upgrade-bits",
                    str(spec["upgrade_bits"]),
                    "--cache-json",
                    str(cache_json),
                ]
            )
        else:
            build_cmd.extend(["--permute-strategy", spec["permute_strategy"]])

        init_log = run_dir / "init.log"
        status = "crash"
        snap_ok = "skipped"
        retention = "discard"

        if run_cmd(build_cmd, init_log) == 0:
            quick_summary = collect_run_summary(run_dir=run_dir)
            quick = quick_summary.get("quick_perplexity")
            screen_fail = baseline_quick != float("inf") and quick is not None and quick > baseline_quick + args.quick_screen_margin

            if not screen_fail:
                ppl_cmd = [
                    sys.executable,
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
                run_cmd(ppl_cmd, run_dir / "perplexity.log")

                snap_cmd = [
                    sys.executable,
                    "scripts/snap_and_test_v2.py",
                    "--checkpoint",
                    str(run_dir / "v2_tightened.pt"),
                    "--fp16",
                    "--no-test",
                    "--output",
                    str(run_dir / "snapped_fp16.pt"),
                ]
                run_cmd(snap_cmd, run_dir / "snap.log")
                snap_ok = "true" if (run_dir / "snapped_fp16.pt").exists() else "false"
            else:
                snap_ok = "skipped"

            summary = collect_run_summary(
                run_dir=run_dir,
                baseline_run_dir=str(baseline_run_dir),
                max_size_growth_pct=args.budget_growth_pct,
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
                retention = "keep" if is_keep else "discard"
                if is_keep:
                    best_full = float(full_ppl)

            append_results_row(
                results_tsv=results_tsv,
                commit=commit,
                summary=summary,
                snap_ok=snap_ok,
                status=status,
                retention=retention,
                change_family=spec["change_family"],
                description=spec["description"],
            )

            cleanup_cmd = [
                sys.executable,
                "aq1_autoresearch/cleanup_run.py",
                "--run-dir",
                str(run_dir),
                "--status",
                "keep" if retention == "keep" else "discard",
            ]
            if retention != "keep":
                cleanup_cmd.extend(["--retain-checkpoint", "none"])
            run_cmd(cleanup_cmd, run_dir / "cleanup.log")
        else:
            summary = collect_run_summary(run_dir=run_dir)
            append_results_row(
                results_tsv=results_tsv,
                commit=commit,
                summary=summary,
                snap_ok=snap_ok,
                status=status,
                retention=retention,
                change_family=spec["change_family"],
                description=spec["description"],
            )

        write_state = {
            "last_run": str(run_dir),
            "last_status": status,
            "best_full_perplexity": None if best_full == float("inf") else best_full,
            "elapsed_hours": (time.time() - started_at) / 3600.0,
        }
        with open(runner_log, "w", encoding="utf-8") as f:
            json.dump(write_state, f, indent=2, sort_keys=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
