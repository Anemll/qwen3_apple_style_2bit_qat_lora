#!/usr/bin/env python3
"""
Google Drive sync utility for Colab training.

Simplifies checkpoint and cache management between Colab local storage and Google Drive.

Usage (in Colab):
    # Mount drive first (if not already)
    from google.colab import drive
    drive.mount('/content/drive')

    # Sync runs (checkpoints)
    !python scripts/gdrive_sync.py up runs/SR-011_mlp_autosnap
    !python scripts/gdrive_sync.py down SR-011_mlp_autosnap
    !python scripts/gdrive_sync.py down SR-011_mlp_autosnap --only "v2_*.pt"
    !python scripts/gdrive_sync.py down runs/SR-011_mlp_autosnap/v2_checkpoint.pt
    !python scripts/gdrive_sync.py list

    # Sync caches (use --cache flag)
    !python scripts/gdrive_sync.py up caches/alpaca_chat_L128 --cache
    !python scripts/gdrive_sync.py down alpaca_chat_L128 --cache
    !python scripts/gdrive_sync.py list --cache

Environment variables:
    GDRIVE_BASE: Base directory for runs (default: /content/drive/MyDrive/qwen3_runs)
    GDRIVE_CACHES: Base directory for caches (default: /content/drive/MyDrive/qwen3_caches)

Training Commands:
    # SR-011 Phase 1: MLP mags and scales (with auto-snap)
    python scripts/train_v2_simple.py \
        --config q4a4_r32 \
        --v2-checkpoint runs/SR-011_q4_a4_r32_from_scratch/model_state_dict.pt \
        --cache-dir caches/alpaca_chat_think_both_L128_K128_R1024 \
        --output-dir runs/SR-011_q4_a4_r32_mlp_autosnap \
        --mlp-only \
        --mixed-precision \
        --max-steps 4000 \
        --batch-size 4 \
        --accumulation-steps 2 \
        --lr 2e-4 \
        --warmup-steps 200 \
        --min-lr-ratio 0.1 \
        --temperature 2.0 \
        --hard-top1 0.2 --hard-top1-end 0.0 \
        --hard-full 5e-05 \
        --clip-grad-norm 1.0 \
        --save-steps 200 \
        --eval-steps 100 \
        --auto-snap-mags \
        --auto-snap-target mlp \
        --auto-snap-threshold 0.05 \
        --auto-snap-patience 2 \
        --auto-snap-start-step 100 \
        --auto-snap-min-saves 2 \
        --anchor-ckpt runs/SR-011_q4_a4_r32_from_scratch/model_state_dict.pt \
        --anchor-kl-weight 0.002 \
        --anchor-interval 10 \
        --anchor-samples 1 \
        --wandb --wandb-project qwen3-qat --wandb-run "SR-011_mlp_autosnap_L128_lr2e-4_anchor" \
        --tpu
"""

import argparse
import fnmatch
import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime


# Default paths for runs
DEFAULT_GDRIVE_RUNS = "/content/drive/MyDrive/qwen3_runs"
DEFAULT_LOCAL_RUNS = "runs"

# Default paths for caches
DEFAULT_GDRIVE_CACHES = "/content/drive/MyDrive/qwen3_caches"
DEFAULT_LOCAL_CACHES = "caches"


def get_paths(is_cache: bool = False):
    """Get Google Drive and local base paths based on mode."""
    if is_cache:
        gdrive = os.environ.get("GDRIVE_CACHES", DEFAULT_GDRIVE_CACHES)
        local = DEFAULT_LOCAL_CACHES
    else:
        gdrive = os.environ.get("GDRIVE_BASE", DEFAULT_GDRIVE_RUNS)
        local = DEFAULT_LOCAL_RUNS
    return gdrive, local


def get_gdrive_base(is_cache: bool = False):
    """Get Google Drive base path from env or default."""
    gdrive, _ = get_paths(is_cache)
    return gdrive


def ensure_drive_mounted(is_cache: bool = False):
    """Check if Google Drive is mounted (Colab-specific)."""
    gdrive_base = get_gdrive_base(is_cache)
    if not os.path.exists("/content/drive"):
        print("ERROR: Google Drive not mounted!")
        print("Run this in Colab first:")
        print("  from google.colab import drive")
        print("  drive.mount('/content/drive')")
        return False
    if not os.path.exists(os.path.dirname(gdrive_base)):
        print(f"ERROR: Drive path not found: {os.path.dirname(gdrive_base)}")
        return False
    return True


def get_run_name(path: str) -> str:
    """Extract run name from path like 'runs/SR-011_foo' -> 'SR-011_foo'."""
    return Path(path).name


def parse_down_path(path: str) -> tuple:
    """
    Parse flexible path format for 'down' command.

    Supports:
      - "SR-011_name" -> (run_name="SR-011_name", only_pattern=None)
      - "runs/SR-011_name" -> (run_name="SR-011_name", only_pattern=None)
      - "runs/SR-011_name/file.pt" -> (run_name="SR-011_name", only_pattern="file.pt")
      - "SR-011_name/file.pt" -> (run_name="SR-011_name", only_pattern="file.pt")

    Returns:
        (run_name, only_pattern) tuple
    """
    # Normalize path
    path = path.strip('/')
    parts = path.split('/')

    # Remove 'runs' prefix if present
    if parts[0] == 'runs':
        parts = parts[1:]

    if len(parts) == 0:
        return None, None
    elif len(parts) == 1:
        # Just run name: "SR-011_name"
        return parts[0], None
    elif len(parts) == 2:
        # Run name + file: "SR-011_name/file.pt"
        return parts[0], parts[1]
    else:
        # Multiple levels: take first as run_name, last as file pattern
        # e.g., "SR-011/subdir/file.pt" -> run_name="SR-011", pattern="file.pt"
        return parts[0], parts[-1]


def list_runs(location: str = "both", is_cache: bool = False):
    """List available runs/caches on local and/or Google Drive."""
    gdrive_base, local_base = get_paths(is_cache)
    item_type = "CACHES" if is_cache else "RUNS"
    file_ext = ".pt" if not is_cache else None  # Caches have subdirs, not just .pt files

    print("=" * 60)
    print(f"AVAILABLE {item_type}")
    print("=" * 60)

    if location in ("local", "both"):
        print(f"\nLocal ({local_base}/):")
        if os.path.exists(local_base):
            items = sorted(os.listdir(local_base))
            for item in items:
                item_path = os.path.join(local_base, item)
                if os.path.isdir(item_path):
                    size = get_dir_size(item_path)
                    if is_cache:
                        # For caches, count files
                        files = sum(1 for _ in Path(item_path).rglob('*') if _.is_file())
                        print(f"  {item:<40} {files:>4} files  {size}")
                    else:
                        # For runs, count checkpoints
                        ckpts = [f for f in os.listdir(item_path) if f.endswith('.pt')]
                        print(f"  {item:<40} {len(ckpts):>3} ckpts  {size}")
        else:
            print(f"  (no local {local_base} directory)")

    if location in ("gdrive", "both"):
        print(f"\nGoogle Drive ({gdrive_base}/):")
        if os.path.exists(gdrive_base):
            items = sorted(os.listdir(gdrive_base))
            for item in items:
                item_path = os.path.join(gdrive_base, item)
                if os.path.isdir(item_path):
                    size = get_dir_size(item_path)
                    if is_cache:
                        files = sum(1 for _ in Path(item_path).rglob('*') if _.is_file())
                        print(f"  {item:<40} {files:>4} files  {size}")
                    else:
                        ckpts = [f for f in os.listdir(item_path) if f.endswith('.pt')]
                        print(f"  {item:<40} {len(ckpts):>3} ckpts  {size}")
        else:
            print(f"  (directory not found: {gdrive_base})")


def get_dir_size(path: str) -> str:
    """Get human-readable directory size."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)

    # Format size
    for unit in ['B', 'KB', 'MB', 'GB']:
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} TB"


def sync_up(local_path: str, run_name: str = None, dry_run: bool = False, is_cache: bool = False, exclude: list = None, only: list = None):
    """
    Sync local run/cache to Google Drive.

    Args:
        local_path: Local directory (e.g., 'runs/SR-011_foo' or 'caches/alpaca_L128')
        run_name: Override name on Drive (default: same as local)
        dry_run: Show what would be copied without copying
        is_cache: If True, sync as cache (recursive); if False, sync as run (flat)
        exclude: List of glob patterns to exclude (e.g., ['checkpoint_step*', '*.tmp'])
        only: List of glob patterns to include (e.g., ['*1200*', 'best*']). If set, only matching files are synced.
    """
    if not ensure_drive_mounted(is_cache):
        return False

    gdrive_base = get_gdrive_base(is_cache)
    item_type = "cache" if is_cache else "run"
    exclude = exclude or []
    only = only or []

    def should_exclude(filename):
        """Check if filename matches any exclude pattern."""
        for pattern in exclude:
            if fnmatch.fnmatch(filename, pattern):
                return True
        return False

    def should_include(filename):
        """Check if filename matches any 'only' pattern. Empty list = include all."""
        if not only:
            return True
        for pattern in only:
            if fnmatch.fnmatch(filename, pattern):
                return True
        return False

    if not os.path.exists(local_path):
        print(f"ERROR: Local path not found: {local_path}")
        return False

    run_name = run_name or get_run_name(local_path)
    gdrive_path = os.path.join(gdrive_base, run_name)

    print("=" * 60)
    print(f"SYNC UP {'CACHE' if is_cache else 'RUN'} (Local -> Google Drive)")
    print("=" * 60)
    print(f"Source:      {local_path}")
    print(f"Destination: {gdrive_path}")
    if only:
        print(f"Only:        {', '.join(only)}")
    if exclude:
        print(f"Exclude:     {', '.join(exclude)}")

    # Get files to sync (recursive for caches)
    local_files = {}  # relative_path -> full_path
    excluded_count = 0
    filtered_count = 0
    if is_cache:
        # Recursive walk for caches
        for root, dirs, files in os.walk(local_path):
            for f in files:
                if not should_include(f):
                    filtered_count += 1
                    continue
                if should_exclude(f):
                    excluded_count += 1
                    continue
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, local_path)
                local_files[rel_path] = full_path
    else:
        # Flat listing for runs
        for f in os.listdir(local_path):
            if os.path.isfile(os.path.join(local_path, f)):
                if not should_include(f):
                    filtered_count += 1
                    continue
                if should_exclude(f):
                    excluded_count += 1
                    continue
                local_files[f] = os.path.join(local_path, f)

    if filtered_count > 0:
        print(f"Filtered:    {filtered_count} files (not matching --only)")
    if excluded_count > 0:
        print(f"Excluded:    {excluded_count} files")

    gdrive_files = {}
    if os.path.exists(gdrive_path):
        if is_cache:
            for root, dirs, files in os.walk(gdrive_path):
                for f in files:
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, gdrive_path)
                    gdrive_files[rel_path] = full_path
        else:
            for f in os.listdir(gdrive_path):
                if os.path.isfile(os.path.join(gdrive_path, f)):
                    gdrive_files[f] = os.path.join(gdrive_path, f)

    # Find new/modified files
    to_copy = []
    for rel_path, local_file in local_files.items():
        gdrive_file = os.path.join(gdrive_path, rel_path)

        if rel_path not in gdrive_files:
            to_copy.append((rel_path, "new"))
        elif os.path.getmtime(local_file) > os.path.getmtime(gdrive_file):
            to_copy.append((rel_path, "modified"))

    if not to_copy:
        print("\nNo files to sync (Drive is up to date)")
        return True

    print(f"\nFiles to sync: {len(to_copy)}")
    for f, status in to_copy:
        size = os.path.getsize(local_files[f])
        size_str = f"{size / 1024 / 1024:.1f} MB" if size > 1024*1024 else f"{size / 1024:.1f} KB"
        print(f"  [{status:8}] {f:<50} {size_str}")

    if dry_run:
        print("\n[DRY RUN] No files copied")
        return True

    # Create destination if needed
    os.makedirs(gdrive_path, exist_ok=True)

    # Copy files
    print("\nCopying...")
    for rel_path, status in to_copy:
        src = local_files[rel_path]
        dst = os.path.join(gdrive_path, rel_path)
        # Create subdirectories if needed (for caches)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print(f"  {rel_path}...", end=" ", flush=True)
        shutil.copy2(src, dst)
        print("done")

    print(f"\nSynced {len(to_copy)} files to Google Drive")
    return True


def sync_down(run_name: str, local_path: str = None, dry_run: bool = False, is_cache: bool = False, only: list = None, exclude: list = None):
    """
    Sync run/cache from Google Drive to local.

    Args:
        run_name: Name on Drive (e.g., 'SR-011_foo' or 'alpaca_L128')
        local_path: Local destination (default: runs/<name> or caches/<name>)
        dry_run: Show what would be copied without copying
        is_cache: If True, sync as cache (recursive); if False, sync as run (flat)
        only: List of glob patterns to include (e.g., ['*1200*', 'best*']). If set, only matching files are synced.
        exclude: List of glob patterns to exclude (e.g., ['checkpoint_step*', '*.tmp'])
    """
    if not ensure_drive_mounted(is_cache):
        return False

    gdrive_base, local_base = get_paths(is_cache)
    gdrive_path = os.path.join(gdrive_base, run_name)
    local_path = local_path or os.path.join(local_base, run_name)
    only = only or []
    exclude = exclude or []

    def should_include(filename):
        """Check if filename matches any 'only' pattern. Empty list = include all."""
        if not only:
            return True
        for pattern in only:
            if fnmatch.fnmatch(filename, pattern):
                return True
        return False

    def should_exclude(filename):
        """Check if filename matches any exclude pattern."""
        for pattern in exclude:
            if fnmatch.fnmatch(filename, pattern):
                return True
        return False

    item_type = "cache" if is_cache else "run"
    if not os.path.exists(gdrive_path):
        print(f"ERROR: {item_type.title()} not found on Drive: {gdrive_path}")
        print(f"\nAvailable {item_type}s:")
        list_runs("gdrive", is_cache)
        return False

    print("=" * 60)
    print(f"SYNC DOWN {'CACHE' if is_cache else 'RUN'} (Google Drive -> Local)")
    print("=" * 60)
    print(f"Source:      {gdrive_path}")
    print(f"Destination: {local_path}")
    if only:
        print(f"Only:        {', '.join(only)}")
    if exclude:
        print(f"Exclude:     {', '.join(exclude)}")

    # Get files to sync (recursive for caches)
    gdrive_files = {}  # relative_path -> full_path
    filtered_count = 0
    excluded_count = 0
    if is_cache:
        for root, dirs, files in os.walk(gdrive_path):
            for f in files:
                if not should_include(f):
                    filtered_count += 1
                    continue
                if should_exclude(f):
                    excluded_count += 1
                    continue
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, gdrive_path)
                gdrive_files[rel_path] = full_path
    else:
        for f in os.listdir(gdrive_path):
            if os.path.isfile(os.path.join(gdrive_path, f)):
                if not should_include(f):
                    filtered_count += 1
                    continue
                if should_exclude(f):
                    excluded_count += 1
                    continue
                gdrive_files[f] = os.path.join(gdrive_path, f)

    if filtered_count > 0:
        print(f"Filtered:    {filtered_count} files (not matching --only)")
    if excluded_count > 0:
        print(f"Excluded:    {excluded_count} files")

    local_files = {}
    if os.path.exists(local_path):
        if is_cache:
            for root, dirs, files in os.walk(local_path):
                for f in files:
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, local_path)
                    local_files[rel_path] = full_path
        else:
            for f in os.listdir(local_path):
                if os.path.isfile(os.path.join(local_path, f)):
                    local_files[f] = os.path.join(local_path, f)

    # Find new/modified files
    to_copy = []
    for rel_path, gdrive_file in gdrive_files.items():
        local_file = os.path.join(local_path, rel_path)

        if rel_path not in local_files:
            to_copy.append((rel_path, "new"))
        elif os.path.getmtime(gdrive_file) > os.path.getmtime(local_file):
            to_copy.append((rel_path, "modified"))

    if not to_copy:
        print("\nNo files to sync (Local is up to date)")
        return True

    print(f"\nFiles to sync: {len(to_copy)}")
    for rel_path, status in to_copy:
        size = os.path.getsize(gdrive_files[rel_path])
        size_str = f"{size / 1024 / 1024:.1f} MB" if size > 1024*1024 else f"{size / 1024:.1f} KB"
        print(f"  [{status:8}] {rel_path:<50} {size_str}")

    if dry_run:
        print("\n[DRY RUN] No files copied")
        return True

    # Create destination if needed
    os.makedirs(local_path, exist_ok=True)

    # Copy files
    print("\nCopying...")
    for rel_path, status in to_copy:
        src = gdrive_files[rel_path]
        dst = os.path.join(local_path, rel_path)
        # Create subdirectories if needed (for caches)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print(f"  {rel_path}...", end=" ", flush=True)
        shutil.copy2(src, dst)
        print("done")

    print(f"\nSynced {len(to_copy)} files from Google Drive")
    return True


def show_status(local_path: str, is_cache: bool = False):
    """Show sync status between local and Drive."""
    if not ensure_drive_mounted(is_cache):
        return

    gdrive_base = get_gdrive_base(is_cache)
    run_name = get_run_name(local_path)
    gdrive_path = os.path.join(gdrive_base, run_name)

    print("=" * 60)
    print(f"SYNC STATUS: {run_name}")
    print("=" * 60)

    local_exists = os.path.exists(local_path)
    gdrive_exists = os.path.exists(gdrive_path)

    print(f"Local:  {local_path} {'(exists)' if local_exists else '(not found)'}")
    print(f"Drive:  {gdrive_path} {'(exists)' if gdrive_exists else '(not found)'}")

    if not local_exists and not gdrive_exists:
        print("\nBoth paths not found!")
        return

    # Collect all files (recursive for caches)
    local_files = {}
    if local_exists:
        if is_cache:
            for root, dirs, files in os.walk(local_path):
                for f in files:
                    fp = os.path.join(root, f)
                    rel_path = os.path.relpath(fp, local_path)
                    local_files[rel_path] = {
                        'mtime': os.path.getmtime(fp),
                        'size': os.path.getsize(fp)
                    }
        else:
            for f in os.listdir(local_path):
                fp = os.path.join(local_path, f)
                if os.path.isfile(fp):
                    local_files[f] = {
                        'mtime': os.path.getmtime(fp),
                        'size': os.path.getsize(fp)
                    }

    gdrive_files = {}
    if gdrive_exists:
        if is_cache:
            for root, dirs, files in os.walk(gdrive_path):
                for f in files:
                    fp = os.path.join(root, f)
                    rel_path = os.path.relpath(fp, gdrive_path)
                    gdrive_files[rel_path] = {
                        'mtime': os.path.getmtime(fp),
                        'size': os.path.getsize(fp)
                    }
        else:
            for f in os.listdir(gdrive_path):
                fp = os.path.join(gdrive_path, f)
                if os.path.isfile(fp):
                    gdrive_files[f] = {
                        'mtime': os.path.getmtime(fp),
                        'size': os.path.getsize(fp)
                    }

    all_files = sorted(set(local_files.keys()) | set(gdrive_files.keys()))

    print(f"\nFiles: {len(all_files)}")
    print("-" * 60)

    in_sync = 0
    local_newer = 0
    drive_newer = 0
    local_only = 0
    drive_only = 0

    for f in all_files:
        in_local = f in local_files
        in_drive = f in gdrive_files

        if in_local and in_drive:
            local_time = local_files[f]['mtime']
            drive_time = gdrive_files[f]['mtime']

            if abs(local_time - drive_time) < 1:  # Within 1 second = same
                status = "in sync"
                in_sync += 1
            elif local_time > drive_time:
                status = "LOCAL newer"
                local_newer += 1
            else:
                status = "DRIVE newer"
                drive_newer += 1
        elif in_local:
            status = "LOCAL only"
            local_only += 1
        else:
            status = "DRIVE only"
            drive_only += 1

        # Only show non-synced files
        if status != "in sync":
            print(f"  [{status:12}] {f}")

    print("-" * 60)
    print(f"In sync:     {in_sync}")
    print(f"Local newer: {local_newer}")
    print(f"Drive newer: {drive_newer}")
    print(f"Local only:  {local_only}")
    print(f"Drive only:  {drive_only}")

    if local_newer > 0 or local_only > 0:
        print(f"\nRun 'sync up' to upload {local_newer + local_only} files")
    if drive_newer > 0 or drive_only > 0:
        print(f"Run 'sync down' to download {drive_newer + drive_only} files")


def find_latest_checkpoint(path: str) -> str:
    """Find the latest checkpoint file in a directory."""
    if not os.path.exists(path):
        return None

    ckpts = []
    for f in os.listdir(path):
        if f.endswith('.pt'):
            fp = os.path.join(path, f)
            ckpts.append((f, os.path.getmtime(fp)))

    if not ckpts:
        return None

    ckpts.sort(key=lambda x: x[1], reverse=True)
    return ckpts[0][0]


def resume_path(run_name: str) -> str:
    """
    Get the checkpoint path to resume training from.

    Checks both local and Drive, returns the latest.
    Syncs from Drive if needed.
    """
    if not ensure_drive_mounted():
        return None

    gdrive_base = get_gdrive_base()
    local_path = os.path.join(DEFAULT_LOCAL_RUNS, run_name)
    gdrive_path = os.path.join(gdrive_base, run_name)

    local_latest = find_latest_checkpoint(local_path)
    gdrive_latest = find_latest_checkpoint(gdrive_path)

    print("=" * 60)
    print(f"RESUME: {run_name}")
    print("=" * 60)

    if local_latest:
        local_time = os.path.getmtime(os.path.join(local_path, local_latest))
        print(f"Local latest:  {local_latest}")
        print(f"  Modified: {datetime.fromtimestamp(local_time)}")
    else:
        print("Local latest:  (none)")

    if gdrive_latest:
        gdrive_time = os.path.getmtime(os.path.join(gdrive_path, gdrive_latest))
        print(f"Drive latest:  {gdrive_latest}")
        print(f"  Modified: {datetime.fromtimestamp(gdrive_time)}")
    else:
        print("Drive latest:  (none)")

    # Determine which to use
    if not local_latest and not gdrive_latest:
        print("\nNo checkpoints found!")
        return None

    if not local_latest:
        # Sync from drive
        print(f"\nSyncing {gdrive_latest} from Drive...")
        sync_down(run_name)
        return os.path.join(local_path, gdrive_latest)

    if not gdrive_latest:
        return os.path.join(local_path, local_latest)

    # Both exist - compare times
    if gdrive_time > local_time:
        print(f"\nDrive is newer - syncing...")
        sync_down(run_name)
        return os.path.join(local_path, gdrive_latest)
    else:
        return os.path.join(local_path, local_latest)


def main():
    parser = argparse.ArgumentParser(
        description='Google Drive sync for Colab training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all runs
  python scripts/gdrive_sync.py list

  # Upload local run to Drive
  python scripts/gdrive_sync.py up runs/SR-011_mlp_autosnap

  # Download run from Drive (multiple formats supported)
  python scripts/gdrive_sync.py down SR-011_mlp_autosnap
  python scripts/gdrive_sync.py down SR-011_mlp_autosnap --only "v2_*.pt"
  python scripts/gdrive_sync.py down runs/SR-011_mlp_autosnap/v2_checkpoint.pt

  # Check sync status
  python scripts/gdrive_sync.py status runs/SR-011_mlp_autosnap

  # Get resume checkpoint path (auto-syncs if needed)
  python scripts/gdrive_sync.py resume SR-011_mlp_autosnap

  # Dry run (show what would be synced)
  python scripts/gdrive_sync.py up runs/SR-011_mlp_autosnap --dry-run

  # Exclude files by pattern (e.g., skip intermediate checkpoints)
  python scripts/gdrive_sync.py up runs/SR-011 --exclude "checkpoint_step*"

  # Sync caches (use --cache flag)
  python scripts/gdrive_sync.py up caches/alpaca_L128 --cache
  python scripts/gdrive_sync.py down alpaca_L128 --cache
  python scripts/gdrive_sync.py list --cache

Environment:
  GDRIVE_BASE: Google Drive base path for runs (default: /content/drive/MyDrive/qwen3_runs)
  GDRIVE_CACHES: Google Drive base path for caches (default: /content/drive/MyDrive/qwen3_caches)
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Command')

    # list
    list_parser = subparsers.add_parser('list', help='List available runs or caches')
    list_parser.add_argument('--location', choices=['local', 'gdrive', 'both'], default='both')
    list_parser.add_argument('--cache', action='store_true', help='List caches instead of runs')

    # up (sync local -> drive)
    up_parser = subparsers.add_parser('up', help='Sync local run/cache to Google Drive')
    up_parser.add_argument('local_path', help='Local directory (e.g., runs/SR-011_foo or caches/alpaca_L128)')
    up_parser.add_argument('--name', help='Override name on Drive')
    up_parser.add_argument('--dry-run', action='store_true', help='Show what would be synced')
    up_parser.add_argument('--cache', action='store_true', help='Sync as cache (recursive) instead of run')
    up_parser.add_argument('--exclude', action='append', default=[],
                          help='Glob pattern to exclude (can be used multiple times, e.g., --exclude "checkpoint_step*")')
    up_parser.add_argument('--only', action='append', default=[],
                          help='Only sync files matching pattern (can be used multiple times, e.g., --only "*1200*")')

    # down (sync drive -> local)
    down_parser = subparsers.add_parser('down', help='Sync run/cache from Google Drive to local')
    down_parser.add_argument('path', help='Run name or path (e.g., SR-011_foo, runs/SR-011_foo/file.pt)')
    down_parser.add_argument('--local', help='Local destination path')
    down_parser.add_argument('--dry-run', action='store_true', help='Show what would be synced')
    down_parser.add_argument('--cache', action='store_true', help='Sync as cache (recursive) instead of run')
    down_parser.add_argument('--only', action='append', default=[],
                          help='Only sync files matching pattern (can be used multiple times, e.g., --only "*1200*")')
    down_parser.add_argument('--exclude', action='append', default=[],
                          help='Glob pattern to exclude (can be used multiple times, e.g., --exclude "checkpoint_step*")')

    # status
    status_parser = subparsers.add_parser('status', help='Show sync status')
    status_parser.add_argument('local_path', help='Local directory')
    status_parser.add_argument('--cache', action='store_true', help='Check cache status instead of run')

    # resume
    resume_parser = subparsers.add_parser('resume', help='Get resume checkpoint path')
    resume_parser.add_argument('run_name', help='Run name')

    args = parser.parse_args()

    if args.command == 'list':
        list_runs(args.location, args.cache)
    elif args.command == 'up':
        sync_up(args.local_path, args.name, args.dry_run, args.cache, args.exclude, args.only)
    elif args.command == 'down':
        # Parse flexible path format: "SR-011_name", "runs/SR-011_name", "runs/SR-011_name/file.pt"
        run_name, path_pattern = parse_down_path(args.path)
        # Combine patterns from path and --only flag
        only_patterns = list(args.only)  # Copy to avoid modifying original
        if path_pattern:
            only_patterns.append(path_pattern)
        sync_down(run_name, args.local, args.dry_run, args.cache, only_patterns if only_patterns else None, args.exclude if args.exclude else None)
    elif args.command == 'status':
        show_status(args.local_path, args.cache)
    elif args.command == 'resume':
        path = resume_path(args.run_name)
        if path:
            print(f"\nResume from: {path}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
