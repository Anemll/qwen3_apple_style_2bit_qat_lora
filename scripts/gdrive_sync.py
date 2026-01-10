#!/usr/bin/env python3
"""
Google Drive sync utility for Colab training.

Simplifies checkpoint management between Colab local storage and Google Drive.

Usage (in Colab):
    # Mount drive first (if not already)
    from google.colab import drive
    drive.mount('/content/drive')

    # Sync commands
    !python scripts/gdrive_sync.py up runs/SR-011_mlp_autosnap
    !python scripts/gdrive_sync.py down SR-011_mlp_autosnap
    !python scripts/gdrive_sync.py list
    !python scripts/gdrive_sync.py status runs/SR-011_mlp_autosnap

Environment variables:
    GDRIVE_BASE: Base directory on Google Drive (default: /content/drive/MyDrive/qwen3_runs)
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime


# Default paths
DEFAULT_GDRIVE_BASE = "/content/drive/MyDrive/qwen3_runs"
DEFAULT_LOCAL_BASE = "runs"


def get_gdrive_base():
    """Get Google Drive base path from env or default."""
    return os.environ.get("GDRIVE_BASE", DEFAULT_GDRIVE_BASE)


def ensure_drive_mounted():
    """Check if Google Drive is mounted (Colab-specific)."""
    gdrive_base = get_gdrive_base()
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


def list_runs(location: str = "both"):
    """List available runs on local and/or Google Drive."""
    gdrive_base = get_gdrive_base()

    print("=" * 60)
    print("AVAILABLE RUNS")
    print("=" * 60)

    if location in ("local", "both"):
        print(f"\nLocal ({DEFAULT_LOCAL_BASE}/):")
        if os.path.exists(DEFAULT_LOCAL_BASE):
            runs = sorted(os.listdir(DEFAULT_LOCAL_BASE))
            for run in runs:
                run_path = os.path.join(DEFAULT_LOCAL_BASE, run)
                if os.path.isdir(run_path):
                    # Count checkpoints
                    ckpts = [f for f in os.listdir(run_path) if f.endswith('.pt')]
                    size = get_dir_size(run_path)
                    print(f"  {run:<40} {len(ckpts):>3} ckpts  {size}")
        else:
            print("  (no local runs directory)")

    if location in ("gdrive", "both"):
        print(f"\nGoogle Drive ({gdrive_base}/):")
        if os.path.exists(gdrive_base):
            runs = sorted(os.listdir(gdrive_base))
            for run in runs:
                run_path = os.path.join(gdrive_base, run)
                if os.path.isdir(run_path):
                    ckpts = [f for f in os.listdir(run_path) if f.endswith('.pt')]
                    size = get_dir_size(run_path)
                    print(f"  {run:<40} {len(ckpts):>3} ckpts  {size}")
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


def sync_up(local_path: str, run_name: str = None, dry_run: bool = False):
    """
    Sync local run to Google Drive.

    Args:
        local_path: Local run directory (e.g., 'runs/SR-011_foo')
        run_name: Override run name on Drive (default: same as local)
        dry_run: Show what would be copied without copying
    """
    if not ensure_drive_mounted():
        return False

    gdrive_base = get_gdrive_base()

    if not os.path.exists(local_path):
        print(f"ERROR: Local path not found: {local_path}")
        return False

    run_name = run_name or get_run_name(local_path)
    gdrive_path = os.path.join(gdrive_base, run_name)

    print("=" * 60)
    print("SYNC UP (Local -> Google Drive)")
    print("=" * 60)
    print(f"Source:      {local_path}")
    print(f"Destination: {gdrive_path}")

    # Get files to sync
    local_files = set()
    for f in os.listdir(local_path):
        if os.path.isfile(os.path.join(local_path, f)):
            local_files.add(f)

    gdrive_files = set()
    if os.path.exists(gdrive_path):
        for f in os.listdir(gdrive_path):
            if os.path.isfile(os.path.join(gdrive_path, f)):
                gdrive_files.add(f)

    # Find new/modified files
    to_copy = []
    for f in local_files:
        local_file = os.path.join(local_path, f)
        gdrive_file = os.path.join(gdrive_path, f)

        if f not in gdrive_files:
            to_copy.append((f, "new"))
        elif os.path.getmtime(local_file) > os.path.getmtime(gdrive_file):
            to_copy.append((f, "modified"))

    if not to_copy:
        print("\nNo files to sync (Drive is up to date)")
        return True

    print(f"\nFiles to sync: {len(to_copy)}")
    for f, status in to_copy:
        size = os.path.getsize(os.path.join(local_path, f))
        size_str = f"{size / 1024 / 1024:.1f} MB" if size > 1024*1024 else f"{size / 1024:.1f} KB"
        print(f"  [{status:8}] {f:<50} {size_str}")

    if dry_run:
        print("\n[DRY RUN] No files copied")
        return True

    # Create destination if needed
    os.makedirs(gdrive_path, exist_ok=True)

    # Copy files
    print("\nCopying...")
    for f, status in to_copy:
        src = os.path.join(local_path, f)
        dst = os.path.join(gdrive_path, f)
        print(f"  {f}...", end=" ", flush=True)
        shutil.copy2(src, dst)
        print("done")

    print(f"\nSynced {len(to_copy)} files to Google Drive")
    return True


def sync_down(run_name: str, local_path: str = None, dry_run: bool = False):
    """
    Sync run from Google Drive to local.

    Args:
        run_name: Run name on Drive (e.g., 'SR-011_foo')
        local_path: Local destination (default: runs/<run_name>)
        dry_run: Show what would be copied without copying
    """
    if not ensure_drive_mounted():
        return False

    gdrive_base = get_gdrive_base()
    gdrive_path = os.path.join(gdrive_base, run_name)
    local_path = local_path or os.path.join(DEFAULT_LOCAL_BASE, run_name)

    if not os.path.exists(gdrive_path):
        print(f"ERROR: Run not found on Drive: {gdrive_path}")
        print("\nAvailable runs:")
        list_runs("gdrive")
        return False

    print("=" * 60)
    print("SYNC DOWN (Google Drive -> Local)")
    print("=" * 60)
    print(f"Source:      {gdrive_path}")
    print(f"Destination: {local_path}")

    # Get files to sync
    gdrive_files = set()
    for f in os.listdir(gdrive_path):
        if os.path.isfile(os.path.join(gdrive_path, f)):
            gdrive_files.add(f)

    local_files = set()
    if os.path.exists(local_path):
        for f in os.listdir(local_path):
            if os.path.isfile(os.path.join(local_path, f)):
                local_files.add(f)

    # Find new/modified files
    to_copy = []
    for f in gdrive_files:
        gdrive_file = os.path.join(gdrive_path, f)
        local_file = os.path.join(local_path, f)

        if f not in local_files:
            to_copy.append((f, "new"))
        elif os.path.getmtime(gdrive_file) > os.path.getmtime(local_file):
            to_copy.append((f, "modified"))

    if not to_copy:
        print("\nNo files to sync (Local is up to date)")
        return True

    print(f"\nFiles to sync: {len(to_copy)}")
    for f, status in to_copy:
        size = os.path.getsize(os.path.join(gdrive_path, f))
        size_str = f"{size / 1024 / 1024:.1f} MB" if size > 1024*1024 else f"{size / 1024:.1f} KB"
        print(f"  [{status:8}] {f:<50} {size_str}")

    if dry_run:
        print("\n[DRY RUN] No files copied")
        return True

    # Create destination if needed
    os.makedirs(local_path, exist_ok=True)

    # Copy files
    print("\nCopying...")
    for f, status in to_copy:
        src = os.path.join(gdrive_path, f)
        dst = os.path.join(local_path, f)
        print(f"  {f}...", end=" ", flush=True)
        shutil.copy2(src, dst)
        print("done")

    print(f"\nSynced {len(to_copy)} files from Google Drive")
    return True


def show_status(local_path: str):
    """Show sync status between local and Drive."""
    if not ensure_drive_mounted():
        return

    gdrive_base = get_gdrive_base()
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

    # Collect all files
    local_files = {}
    if local_exists:
        for f in os.listdir(local_path):
            fp = os.path.join(local_path, f)
            if os.path.isfile(fp):
                local_files[f] = {
                    'mtime': os.path.getmtime(fp),
                    'size': os.path.getsize(fp)
                }

    gdrive_files = {}
    if gdrive_exists:
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
    local_path = os.path.join(DEFAULT_LOCAL_BASE, run_name)
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

  # Download run from Drive
  python scripts/gdrive_sync.py down SR-011_mlp_autosnap

  # Check sync status
  python scripts/gdrive_sync.py status runs/SR-011_mlp_autosnap

  # Get resume checkpoint path (auto-syncs if needed)
  python scripts/gdrive_sync.py resume SR-011_mlp_autosnap

  # Dry run (show what would be synced)
  python scripts/gdrive_sync.py up runs/SR-011_mlp_autosnap --dry-run

Environment:
  GDRIVE_BASE: Google Drive base path (default: /content/drive/MyDrive/qwen3_runs)
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Command')

    # list
    list_parser = subparsers.add_parser('list', help='List available runs')
    list_parser.add_argument('--location', choices=['local', 'gdrive', 'both'], default='both')

    # up (sync local -> drive)
    up_parser = subparsers.add_parser('up', help='Sync local run to Google Drive')
    up_parser.add_argument('local_path', help='Local run directory (e.g., runs/SR-011_foo)')
    up_parser.add_argument('--name', help='Override run name on Drive')
    up_parser.add_argument('--dry-run', action='store_true', help='Show what would be synced')

    # down (sync drive -> local)
    down_parser = subparsers.add_parser('down', help='Sync run from Google Drive to local')
    down_parser.add_argument('run_name', help='Run name on Drive (e.g., SR-011_foo)')
    down_parser.add_argument('--local', help='Local destination path')
    down_parser.add_argument('--dry-run', action='store_true', help='Show what would be synced')

    # status
    status_parser = subparsers.add_parser('status', help='Show sync status')
    status_parser.add_argument('local_path', help='Local run directory')

    # resume
    resume_parser = subparsers.add_parser('resume', help='Get resume checkpoint path')
    resume_parser.add_argument('run_name', help='Run name')

    args = parser.parse_args()

    if args.command == 'list':
        list_runs(args.location)
    elif args.command == 'up':
        sync_up(args.local_path, args.name, args.dry_run)
    elif args.command == 'down':
        sync_down(args.run_name, args.local, args.dry_run)
    elif args.command == 'status':
        show_status(args.local_path)
    elif args.command == 'resume':
        path = resume_path(args.run_name)
        if path:
            print(f"\nResume from: {path}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
