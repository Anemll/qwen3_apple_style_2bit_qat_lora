#!/usr/bin/env python3
"""Prune bulky AQ1 run artifacts after scoring while keeping reproducibility metadata."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
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


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_campaign_metadata(run_dir: Path, repo_root: Path) -> dict[str, str]:
    campaign_path = run_dir / "campaign_snapshot.md"
    if not campaign_path.exists():
        campaign_path = repo_root / "aq1_autoresearch" / "campaign.md"
    metadata: dict[str, str] = {}
    if not campaign_path.exists():
        return metadata
    bullet = re.compile(r"^- (?P<key>[^:]+): (?P<value>.+)$")
    with open(campaign_path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            match = bullet.match(line)
            if not match:
                continue
            key = match.group("key").strip().lower().replace(" ", "_")
            value = match.group("value").strip()
            if value.startswith("`") and value.endswith("`"):
                value = value[1:-1]
            metadata[key] = value
    return metadata


def extract_logged_command(log_path: Path) -> list[str] | None:
    if not log_path.exists():
        return None
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("$ "):
                try:
                    return shlex.split(line[2:].strip())
                except ValueError:
                    return None
    return None


def read_group_size_from_init_log(log_path: Path) -> str | None:
    if not log_path.exists():
        return None
    pattern = re.compile(r"^\s*Group size:\s*(?P<value>\d+)\s*$")
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            match = pattern.match(line.rstrip("\n"))
            if match:
                return match.group("value")
    return None


def rewrite_command_arg(arg: str, repo_root: Path, run_dir: Path) -> str:
    repo_root_str = str(repo_root.resolve())
    run_dir_str = str(run_dir.resolve())
    python_bin = str(repo_root / ".venv" / "bin" / "python")
    if arg == python_bin:
        return "{PYTHON_BIN}"
    if arg.startswith(run_dir_str):
        return arg.replace(run_dir_str, "{RUN_DIR}", 1)
    if arg.startswith(repo_root_str):
        return arg.replace(repo_root_str, "{REPO_ROOT}", 1)
    return arg


def render_shell_arg(arg: str) -> str:
    if arg == "{PYTHON_BIN}":
        return '"${PYTHON_BIN}"'
    if "{REPO_ROOT}" in arg or "{RUN_DIR}" in arg:
        rendered = arg.replace("{REPO_ROOT}", "${REPO_ROOT}").replace("{RUN_DIR}", "${RUN_DIR}")
        rendered = rendered.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{rendered}"'
    return shlex.quote(arg)


def render_shell_command(argv: list[str], repo_root: Path, run_dir: Path) -> str:
    return " ".join(render_shell_arg(rewrite_command_arg(arg, repo_root=repo_root, run_dir=run_dir)) for arg in argv)


def synthesize_init_command(run_dir: Path, repo_root: Path, campaign: dict[str, str]) -> list[str] | None:
    metrics = load_json(run_dir / "init_metrics.json") or {}
    model_id = metrics.get("model_id") or campaign.get("model_id")
    preset = metrics.get("preset") or campaign.get("config_preset")
    if not model_id or not preset:
        return None

    cmd = [
        str(repo_root / ".venv" / "bin" / "python"),
        "scripts/init_model_v2.py",
        "--output",
        str(run_dir),
        "--config",
        str(preset),
    ]
    group_size = read_group_size_from_init_log(run_dir / "init.log")
    if group_size:
        cmd.extend(["--group-size", group_size])

    steps = metrics.get("steps") or {}
    if "lut_search" in steps:
        cmd.append("--search-lut")
    if "group_search" in steps:
        tested = steps["group_search"].get("group_sizes_tested") or []
        if tested:
            cmd.extend(["--group-search", ",".join(str(value) for value in tested)])

    cmd.append("--ppl")
    quick_chunks = campaign.get("quick_ppl_chunks")
    if quick_chunks:
        cmd.extend(["--ppl-chunks", quick_chunks])
    return cmd


def synthesize_ppl_command(run_dir: Path, repo_root: Path, campaign: dict[str, str]) -> list[str] | None:
    if not (run_dir / "config.json").exists():
        return None
    checkpoint = run_dir / "v2_tightened.pt"
    cmd = [
        str(repo_root / ".venv" / "bin" / "python"),
        "scripts/measure_perplexity.py",
        str(checkpoint),
        "--config",
        str(run_dir / "config.json"),
        "--device",
        campaign.get("device", "mps"),
        "--dtype",
        campaign.get("dtype_for_ppl", "fp16"),
    ]
    full_chunks = campaign.get("full_ppl_chunks")
    if full_chunks:
        cmd.extend(["--max-chunks", full_chunks])
    return cmd


def synthesize_snap_command(run_dir: Path, repo_root: Path) -> list[str]:
    return [
        str(repo_root / ".venv" / "bin" / "python"),
        "scripts/snap_and_test_v2.py",
        "--checkpoint",
        str(run_dir / "v2_tightened.pt"),
        "--fp16",
        "--no-test",
        "--output",
        str(run_dir / "snapped_fp16.pt"),
    ]


def write_reproduce_script(run_dir: Path, repo_root: Path, summary: dict[str, Any], status: str, dry_run: bool) -> Path | None:
    campaign = load_campaign_metadata(run_dir, repo_root=repo_root)
    stages = [
        ("init.log", "build/init"),
        ("perplexity.log", "full perplexity"),
        ("snap.log", "snap"),
    ]
    rendered_commands: list[tuple[str, str]] = []
    for log_name, label in stages:
        argv = extract_logged_command(run_dir / log_name)
        if argv is None:
            if log_name == "init.log":
                argv = synthesize_init_command(run_dir, repo_root=repo_root, campaign=campaign)
            elif log_name == "perplexity.log":
                argv = synthesize_ppl_command(run_dir, repo_root=repo_root, campaign=campaign)
            elif log_name == "snap.log":
                argv = synthesize_snap_command(run_dir, repo_root=repo_root)
        if argv:
            rendered_commands.append((label, render_shell_command(argv, repo_root=repo_root, run_dir=run_dir)))

    if not rendered_commands:
        return None

    reproduce_path = run_dir / "reproduce.sh"
    if dry_run:
        return reproduce_path

    full_ppl = summary.get("full_perplexity")
    quick_ppl = summary.get("quick_perplexity")
    score = summary.get("score")
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'SEARCH_DIR="$SCRIPT_DIR"',
        'REPO_ROOT=""',
        'while [[ "$SEARCH_DIR" != "/" ]]; do',
        '  if [[ -d "$SEARCH_DIR/aq1_autoresearch" && -d "$SEARCH_DIR/scripts" ]]; then',
        '    REPO_ROOT="$SEARCH_DIR"',
        "    break",
        "  fi",
        '  SEARCH_DIR="$(dirname "$SEARCH_DIR")"',
        "done",
        'if [[ -z "$REPO_ROOT" ]]; then',
        '  echo "error: could not find repo root from $SCRIPT_DIR" >&2',
        "  exit 1",
        "fi",
        'RUN_DIR="$SCRIPT_DIR"',
        'PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"',
        'if [[ ! -x "$PYTHON_BIN" ]]; then',
        '  echo "warning: ${PYTHON_BIN} not found; falling back to python3" >&2',
        '  PYTHON_BIN="python3"',
        "fi",
        'cd "$REPO_ROOT"',
        "",
        f'echo "reproducing $(basename "$RUN_DIR")"',
        f'echo "recorded status: {status}"',
        f'echo "recorded score: {score if score is not None else "-"}"',
        f'echo "recorded quick perplexity: {quick_ppl if quick_ppl is not None else "-"}"',
        f'echo "recorded full perplexity: {full_ppl if full_ppl is not None else "-"}"',
        "",
    ]
    for label, command in rendered_commands:
        lines.append(f'echo "[{label}]"')
        lines.append(command)
        lines.append("")

    reproduce_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    os.chmod(reproduce_path, 0o755)
    return reproduce_path


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
    reproduce_script = write_reproduce_script(run_dir, repo_root=repo_root, summary=summary, status=args.status, dry_run=args.dry_run)
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
        "reproduce_script": str(reproduce_script) if reproduce_script else None,
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
