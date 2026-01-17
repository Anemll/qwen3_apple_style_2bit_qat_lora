#!/usr/bin/env python3
"""
Continuous sync loop for training checkpoints to Google Drive.

Periodically syncs multiple folders using gdrive_sync.py.
Designed for Colab to protect against disconnects.

Usage:
    # Sync runs and lut_candidates every 60 seconds
    python scripts/sync_loop.py runs/SR-012_luts lut_candidates

    # Custom interval (30 seconds)
    python scripts/sync_loop.py runs/SR-012_luts lut_candidates --interval 30

    # Run in background (Colab)
    !nohup python scripts/sync_loop.py runs/myrun lut_candidates > /tmp/sync_loop.log 2>&1 &
    !echo "Sync loop PID: $(pgrep -f sync_loop.py)"

    # With exclusions
    python scripts/sync_loop.py runs/myrun --exclude "checkpoint_step*"

    # Dry run (show what would sync)
    python scripts/sync_loop.py runs/myrun lut_candidates --dry-run
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_DIR = Path(__file__).parent.parent


def sync_folder(folder: str, dry_run: bool = False, exclude: list = None, size_only: bool = False):
    """Sync a single folder using gdrive_sync.py up."""
    cmd = [
        sys.executable,
        str(REPO_DIR / 'scripts' / 'gdrive_sync.py'),
        'up',
        folder,
        '--size-only' if size_only else '',
    ]
    # Remove empty strings
    cmd = [c for c in cmd if c]

    if dry_run:
        cmd.append('--dry-run')

    if exclude:
        for pattern in exclude:
            cmd.extend(['--exclude', pattern])

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


def format_time():
    """Format current time for logging."""
    return datetime.now().strftime('%H:%M:%S')


def main():
    parser = argparse.ArgumentParser(
        description='Continuous sync loop for training checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('folders', nargs='+',
                        help='Folders to sync (e.g., runs/SR-012 lut_candidates)')
    parser.add_argument('-i', '--interval', type=int, default=60,
                        help='Sync interval in seconds (default: 60)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be synced without copying')
    parser.add_argument('--exclude', action='append', default=[],
                        help='Glob pattern to exclude (can be repeated)')
    parser.add_argument('--size-only', action='store_true',
                        help='Compare by size only (faster for write-once files)')
    parser.add_argument('--once', action='store_true',
                        help='Run once and exit (no loop)')

    args = parser.parse_args()

    # Header
    print("=" * 60)
    print("SYNC LOOP - Continuous Google Drive Backup")
    print("=" * 60)
    print(f"Folders:  {', '.join(args.folders)}")
    print(f"Interval: {args.interval}s")
    print(f"PID:      {os.getpid()}")
    if args.exclude:
        print(f"Exclude:  {', '.join(args.exclude)}")
    if args.dry_run:
        print("Mode:     DRY RUN")
    print("=" * 60)
    print()

    # Track stats
    sync_count = 0
    error_count = 0
    running = True

    def handle_signal(signum, frame):
        nonlocal running
        print(f"\n[{format_time()}] Received signal {signum}, stopping...")
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Initial sync
    print(f"[{format_time()}] Initial sync...")
    for folder in args.folders:
        if not os.path.exists(folder):
            print(f"  WARNING: {folder} does not exist (will sync when created)")
            continue

        print(f"  Syncing: {folder}")
        result = sync_folder(folder, args.dry_run, args.exclude, args.size_only)

        # Parse output for summary
        lines = result.stdout.strip().split('\n') if result.stdout else []
        for line in lines:
            if 'Files to sync:' in line or 'No files to sync' in line:
                print(f"    {line.strip()}")
            elif '[new' in line.lower() or '[modified' in line.lower():
                print(f"    {line.strip()}")

        if result.returncode != 0:
            error_count += 1
            if result.stderr:
                print(f"    ERROR: {result.stderr[:200]}")
        else:
            sync_count += 1

    if args.once:
        print(f"\n[{format_time()}] Single run complete. Synced {sync_count} folders.")
        return 0

    # Main loop
    print(f"\n[{format_time()}] Watching for changes every {args.interval}s...")
    print("Press Ctrl+C to stop\n")

    while running:
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            break

        if not running:
            break

        # Sync all folders
        for folder in args.folders:
            if not os.path.exists(folder):
                continue

            result = sync_folder(folder, args.dry_run, args.exclude, args.size_only)

            # Only print if there were changes
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                has_changes = any('Files to sync:' in line for line in lines)
                if has_changes:
                    print(f"[{format_time()}] {folder}:")
                    for line in lines:
                        if 'Files to sync:' in line:
                            print(f"  {line.strip()}")
                        elif '[new' in line.lower() or '[modified' in line.lower():
                            print(f"  {line.strip()}")
                        elif 'Synced' in line and 'files' in line:
                            print(f"  {line.strip()}")
                    sync_count += 1

            if result.returncode != 0:
                error_count += 1

    # Final sync
    print(f"\n[{format_time()}] Final sync before exit...")
    for folder in args.folders:
        if os.path.exists(folder):
            result = sync_folder(folder, args.dry_run, args.exclude, args.size_only)
            if 'Files to sync:' in (result.stdout or ''):
                print(f"  {folder}: synced")

    print(f"\n[{format_time()}] Done. Total syncs: {sync_count}, Errors: {error_count}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
