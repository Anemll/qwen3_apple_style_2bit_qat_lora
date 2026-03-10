#!/usr/bin/env python3
"""Prune bulky AQ1 run artifacts after scoring while keeping reproducibility metadata."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from score_run import collect_run_summary


HEAVY_SUFFIXES = {
    ".pt",
    ".pth",
    ".bin",
    ".safetensors",
    ".npz",
    ".npy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune bulky AQ1 run artifacts while keeping metadata")
    parser.add_argument("--run-dir", required=True, help="Run directory to prune")
    parser.add_argument("--status", choices=["keep", "discard", "crash"], required=True, help="Experiment outcome")
    parser.add_argument(
        "--retain-checkpoint",
        choices=["auto", "none"],
        default="auto",
        help="Retention mode for the selected checkpoint",
    )
    parser.add_argument("--checkpoint", default=None, help="Preferred checkpoint path")
    parser.add_argument("--ppl-log", default=None, help="Optional perplexity log override")
    parser.add_argument("--results-json", default=None, help="Optional results/perplexity.json override")
    parser.add_argument(
        "--remove-dir",
        action="append",
        default=[],
        help="Directory name to remove recursively after pruning heavy files. Can be passed multiple times.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    return parser.parse_args()


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def gather_git_metadata(repo_root: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for label, cmd in {
        "commit": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
    }.items():
        try:
            result = subprocess.run(
                cmd,
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
            )
            value = result.stdout.strip()
            meta[label] = value or None
        except Exception:
            meta[label] = None
    return meta


def should_retain_checkpoint(args: argparse.Namespace, summary: dict[str, Any]) -> Path | None:
    checkpoint = summary.get("candidate_checkpoint")
    if not checkpoint:
        return None
    if args.retain_checkpoint == "none":
        return None
    if args.status == "keep":
        return Path(checkpoint)
    return None


def path_contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def snapshot_campaign(run_dir: Path, repo_root: Path, write_file: bool) -> Path | None:
    campaign = repo_root / "aq1_autoresearch" / "campaign.md"
    if not campaign.exists():
        return None
    snapshot = run_dir / "campaign_snapshot.md"
    if write_file and not snapshot.exists():
        shutil.copyfile(campaign, snapshot)
    return snapshot


def write_metadata_files(run_dir: Path, manifest: dict[str, Any], score_summary: dict[str, Any]) -> None:
    with open(run_dir / "score.json", "w", encoding="utf-8") as f:
        json.dump(score_summary, f, indent=2, sort_keys=True)
    with open(run_dir / "cleanup_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def collect_heavy_files(run_dir: Path, retained_checkpoint: Path | None) -> list[Path]:
    files: list[Path] = []
    retained = retained_checkpoint.resolve() if retained_checkpoint else None
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        if retained is not None and path.resolve() == retained:
            continue
        if path.name in {"score.json", "cleanup_manifest.json", "campaign_snapshot.md"}:
            continue
        if path.suffix.lower() in HEAVY_SUFFIXES:
            files.append(path)
    return sorted(files)


def collect_named_dirs(run_dir: Path, names: set[str], retained_checkpoint: Path | None) -> list[Path]:
    dirs: list[Path] = []
    retained = retained_checkpoint.resolve() if retained_checkpoint else None
    for path in run_dir.rglob("*"):
        if not path.is_dir():
            continue
        if path.name not in names:
            continue
        if retained is not None and path_contains(path.resolve(), retained):
            continue
        dirs.append(path)
    return sorted(dirs, reverse=True)


def remove_empty_dirs(run_dir: Path) -> None:
    for path in sorted((p for p in run_dir.rglob("*") if p.is_dir()), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    repo_root = Path(__file__).resolve().parent.parent

    if not run_dir.exists():
        raise SystemExit(f"missing run dir: {run_dir}")

    summary = collect_run_summary(
        run_dir=run_dir,
        checkpoint=args.checkpoint,
        ppl_log=args.ppl_log,
        results_json=args.results_json,
    )
    retained_checkpoint = should_retain_checkpoint(args, summary)

    run_dir.mkdir(parents=True, exist_ok=True)
    campaign_snapshot = snapshot_campaign(run_dir, repo_root, write_file=not args.dry_run)
    git_meta = gather_git_metadata(repo_root)

    remove_dir_names = {"lut_candidates", "__pycache__"}
    remove_dir_names.update(name for name in args.remove_dir if name)

    heavy_files = collect_heavy_files(run_dir, retained_checkpoint)
    named_dirs = collect_named_dirs(run_dir, remove_dir_names, retained_checkpoint)
    bytes_to_delete = sum(file_size(path) for path in heavy_files)
    bytes_in_named_dirs = 0
    for directory in named_dirs:
        for path in directory.rglob("*"):
            if path.is_file():
                bytes_in_named_dirs += file_size(path)

    manifest = {
        "run_dir": str(run_dir),
        "status": args.status,
        "retention_mode": args.retain_checkpoint,
        "retained_checkpoint": str(retained_checkpoint) if retained_checkpoint else None,
        "campaign_snapshot": str(campaign_snapshot) if campaign_snapshot else None,
        "git": git_meta,
        "score_summary": summary,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "heavy_file_count": len(heavy_files),
        "named_dir_count": len(named_dirs),
        "bytes_to_delete_estimate": bytes_to_delete,
        "bytes_in_named_dirs_estimate": bytes_in_named_dirs,
        "heavy_files": [str(path) for path in heavy_files],
        "named_dirs": [str(path) for path in named_dirs],
        "dry_run": args.dry_run,
    }

    deleted_files: list[str] = []
    deleted_dirs: list[str] = []
    if not args.dry_run:
        write_metadata_files(run_dir, manifest, summary)

        for path in heavy_files:
            if path.exists():
                os.remove(path)
                deleted_files.append(str(path))

        for path in named_dirs:
            if path.exists():
                shutil.rmtree(path)
                deleted_dirs.append(str(path))

        remove_empty_dirs(run_dir)
        manifest["deleted_files"] = deleted_files
        manifest["deleted_dirs"] = deleted_dirs
        manifest["deleted_file_count"] = len(deleted_files)
        manifest["deleted_dir_count"] = len(deleted_dirs)
        write_metadata_files(run_dir, manifest, summary)

    result = {
        "run_dir": str(run_dir),
        "status": args.status,
        "retained_checkpoint": str(retained_checkpoint) if retained_checkpoint else None,
        "heavy_file_count": len(heavy_files),
        "named_dir_count": len(named_dirs),
        "bytes_to_delete_estimate": bytes_to_delete,
        "bytes_in_named_dirs_estimate": bytes_in_named_dirs,
        "dry_run": args.dry_run,
        "deleted_file_count": len(deleted_files),
        "deleted_dir_count": len(deleted_dirs),
    }

    if args.format == "json":
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"run_dir:                 {result['run_dir']}")
        print(f"status:                  {result['status']}")
        print(f"retained_checkpoint:     {result['retained_checkpoint'] or '-'}")
        print(f"heavy_file_count:        {result['heavy_file_count']}")
        print(f"named_dir_count:         {result['named_dir_count']}")
        print(f"bytes_to_delete_estimate:{result['bytes_to_delete_estimate']}")
        print(f"bytes_in_named_dirs:     {result['bytes_in_named_dirs_estimate']}")
        print(f"dry_run:                 {result['dry_run']}")
        print(f"deleted_file_count:      {result['deleted_file_count']}")
        print(f"deleted_dir_count:       {result['deleted_dir_count']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
