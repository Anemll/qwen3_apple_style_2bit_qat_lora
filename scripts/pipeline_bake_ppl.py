#!/usr/bin/env python3
"""
Pipeline: Download → Bake → Perplexity for checkpoints in a run.

Features:
- Idempotent: uses ppl_state.json to skip already-processed steps
- Prefetch: downloads next checkpoint while baking/evaluating current
- Atomic state updates: safe to interrupt and resume
- --best: process best_state_dict.pt with md5-based deduplication

Usage:
    # Basic - process all checkpoints in a run
    python scripts/pipeline_bake_ppl.py srLUT-004b_all_alpaca

    # Or use -b flag
    python scripts/pipeline_bake_ppl.py -b srLUT-004b_all_alpaca

    # With config preset
    python scripts/pipeline_bake_ppl.py srLUT-004b --config q2a4 --dtype fp16

    # Limit to first N checkpoints
    python scripts/pipeline_bake_ppl.py srLUT-004b --max-steps 5

    # Use local runs directory (skip download)
    python scripts/pipeline_bake_ppl.py srLUT-004b --local-only

    # Fast screening with limited chunks
    python scripts/pipeline_bake_ppl.py srLUT-004b --max-chunks 20

    # View summary only
    python scripts/pipeline_bake_ppl.py srLUT-004b --summary

    # Process best_state_dict.pt (force download, md5 dedup)
    python scripts/pipeline_bake_ppl.py srLUT-004b --best --config q4_r32 --device tpu

    # macOS (auto-detects Google Drive location)
    python scripts/pipeline_bake_ppl.py srLUT-004b

State file: runs/<run-name>/ppl_state.json
Baked files: runs/<run-name>/baked_step{N}.pt
Best baked files: runs/<run-name>/baked_best_{md5_short}.pt
"""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

REPO_ROOT = Path(__file__).parent.parent


def find_gdrive_root() -> Optional[Path]:
    """Find Google Drive root based on platform."""
    system = platform.system()
    if system == "Darwin":
        # macOS: ~/Library/CloudStorage/GoogleDrive-*/My Drive
        cloud_storage = Path.home() / "Library" / "CloudStorage"
        if cloud_storage.exists():
            for item in cloud_storage.iterdir():
                if item.name.startswith("GoogleDrive"):
                    my_drive = item / "My Drive"
                    if my_drive.exists():
                        return my_drive
        return None
    else:
        # Colab / Linux
        drive_path = Path("/content/drive/MyDrive")
        if drive_path.exists():
            return drive_path
        return None


def atomic_write_json(path: Path, obj: dict):
    """Write JSON atomically using rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def load_state(path: Path) -> dict:
    """Load state from JSON file."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"[warn] Corrupt state file, starting fresh: {path}")
            return {"version": 1, "run": None, "entries": {}}
    return {"version": 1, "run": None, "entries": {}}


def is_done(state: dict, step: int) -> bool:
    """Check if step is already completed."""
    e = state["entries"].get(str(step))
    return bool(e and e.get("ppl_ok"))


def is_baked(state: dict, step: int) -> bool:
    """Check if step is already baked."""


def load_ppl_cache() -> dict:
    """Load the perplexity results cache from results/perplexity.json."""
    cache_path = REPO_ROOT / "results" / "perplexity.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def get_cached_ppl(run_name: str, step: int, ppl_cache: dict, state: dict) -> Optional[dict]:
    """Check if perplexity result exists in cache. Returns dict with ppl, xe, tokens or None.

    Checks in order:
    1. Per-run ppl_state.json (already loaded as 'state')
    2. Global results/perplexity.json (passed as 'ppl_cache')
    """
    # First check per-run state (highest priority - synced with GDrive)
    step_str = str(step)
    if step_str in state.get("entries", {}):
        entry = state["entries"][step_str]
        if entry.get("ppl_ok") and entry.get("ppl") is not None:
            return {
                "ppl": entry["ppl"],
                "xe": entry.get("xe", 0.0),
                "tokens": entry.get("tokens", 0),
            }

    # Fall back to global cache
    key = f"{run_name}/baked_step{step}.pt"
    if key in ppl_cache:
        entry = ppl_cache[key]
        if "perplexity" in entry:
            return {
                "ppl": entry["perplexity"],
                "xe": entry.get("cross_entropy", 0.0),
                "tokens": entry.get("tokens", 0),
            }
    return None


def sync_state_from_gdrive(run_name: str) -> bool:
    """Try to sync ppl_state.json from Google Drive."""
    state_file = "ppl_state.json"
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "gdrive_sync.py"),
        "down", run_name, "--only", state_file
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            return True
    except (subprocess.TimeoutExpired, Exception):
        pass
    return False


def sync_state_to_gdrive(run_name: str) -> bool:
    """Sync ppl_state.json to Google Drive."""
    state_file = "ppl_state.json"
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "gdrive_sync.py"),
        "up", f"runs/{run_name}", "--only", state_file, "--size-only"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False


def upload_file_to_gdrive(local_path: Path, run_name: str, timeout: int = 600) -> bool:
    """Upload a single file to Google Drive run folder."""
    if not local_path.exists():
        print(f"  [upload] File not found: {local_path}")
        return False

    file_name = local_path.name
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "gdrive_sync.py"),
        "up", f"runs/{run_name}", "--only", file_name
    ]
    try:
        print(f"  [upload] Uploading {file_name} to GDrive ({local_path.stat().st_size / 1e6:.1f} MB)...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            print(f"  [upload] Done: {file_name}")
            return True
        else:
            print(f"  [upload] Failed: {result.stderr[-500:] if result.stderr else 'unknown'}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  [upload] Timed out uploading {file_name}")
        return False
    except Exception as e:
        print(f"  [upload] Error: {e}")
        return False


def find_best_ppl(state: dict, exclude_source: Optional[str] = None) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """Find the best (lowest) PPL from all entries.

    Args:
        state: Pipeline state dict
        exclude_source: Optional source to exclude from comparison (e.g., "step_100" or "best_abc12")

    Returns:
        Tuple of (best_ppl, best_source, best_baked_path)
        - best_ppl: The lowest PPL value, or None if no PPL recorded
        - best_source: "step_N" or "best_XXXXX" identifying the source
        - best_baked_path: Path to the baked checkpoint with best PPL
    """
    best_ppl = float('inf')
    best_source = None
    best_baked_path = None

    # Check step-based entries
    for step_str, entry in state.get("entries", {}).items():
        source = f"step_{step_str}"
        if source == exclude_source:
            continue
        if entry.get("ppl_ok") and entry.get("ppl") is not None:
            ppl = entry["ppl"]
            if ppl < best_ppl:
                best_ppl = ppl
                best_source = source
                best_baked_path = entry.get("baked_path")

    # Check best_state_dict entries
    for md5_key, entry in state.get("best_entries", {}).items():
        source = f"best_{md5_key}"
        if source == exclude_source:
            continue
        if entry.get("ppl_ok") and entry.get("ppl") is not None:
            ppl = entry["ppl"]
            if ppl < best_ppl:
                best_ppl = ppl
                best_source = source
                best_baked_path = entry.get("baked_path")

    if best_ppl == float('inf'):
        return None, None, None

    return best_ppl, best_source, best_baked_path


def snap_checkpoint(baked_path: Path, config: str, output_path: Path, timeout: int = 300) -> bool:
    """Snap a baked checkpoint to FP16 for ANE export.

    Args:
        baked_path: Path to the baked checkpoint
        config: Config preset (q4_r32, q2a4, etc.)
        output_path: Path to save the snapped checkpoint
        timeout: Timeout in seconds

    Returns:
        True if snapping succeeded, False otherwise
    """
    if not baked_path.exists():
        print(f"  [snap] Baked checkpoint not found: {baked_path}")
        return False

    print(f"  [snap] Snapping {baked_path.name} → {output_path.name}...")
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "snap_and_test_v2.py"),
        "--checkpoint", str(baked_path),
        "--config", config,
        "--fp16",
        "--no-test",
        "--output", str(output_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            print(f"  [snap] Failed: {result.stderr[-500:] if result.stderr else 'unknown'}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  [snap] Timed out")
        return False
    except Exception as e:
        print(f"  [snap] Error: {e}")
        return False

    if not output_path.exists():
        print(f"  [snap] Output file not created")
        return False

    print(f"  [snap] Done: {output_path.name} ({output_path.stat().st_size / 1e6:.1f} MB)")
    return True


def check_and_snap_best(
    state: dict,
    current_ppl: float,
    current_source: str,
    baked_path: Path,
    local_dir: Path,
    run_name: str,
    config: str,
    upload: bool = True,
) -> bool:
    """Check if current PPL is the best, and if so, snap and upload.

    Args:
        state: Pipeline state dict
        current_ppl: The PPL just measured
        current_source: Source identifier (e.g., "step_100" or "best_abc12")
        baked_path: Path to the baked checkpoint
        local_dir: Local run directory
        run_name: Run name for GDrive upload
        config: Config preset for snapping
        upload: Whether to upload to GDrive

    Returns:
        True if this was the new best and was snapped/uploaded
    """
    # Find previous best (exclude current source since it's already in state)
    prev_best_ppl, prev_best_source, _ = find_best_ppl(state, exclude_source=current_source)

    # Check if current is the new best
    is_new_best = prev_best_ppl is None or current_ppl < prev_best_ppl

    if not is_new_best:
        if prev_best_ppl is not None:
            print(f"  [best] Not the best: {current_ppl:.2f} > {prev_best_ppl:.2f} ({prev_best_source})")
        return False

    print(f"\n  {'='*50}")
    was_str = f"{prev_best_ppl:.2f}" if prev_best_ppl is not None else "N/A"
    print(f"  NEW BEST PPL: {current_ppl:.2f} (was {was_str})")
    print(f"  {'='*50}")

    # Snap the checkpoint
    snapped_name = f"snapped_best_{current_source.replace('step_', 's')}.pt"
    snapped_path = local_dir / snapped_name

    if not snap_checkpoint(baked_path, config, snapped_path):
        print(f"  [best] Snapping failed, skipping upload")
        return False

    # Update state with snapped info
    state.setdefault("snapped_best", {})
    state["snapped_best"] = {
        "source": current_source,
        "ppl": current_ppl,
        "baked_path": str(baked_path),
        "snapped_path": str(snapped_path),
        "snapped_name": snapped_name,
        "snap_time": datetime.now().isoformat(),
    }

    # Upload to GDrive
    if upload:
        if upload_file_to_gdrive(snapped_path, run_name):
            state["snapped_best"]["uploaded"] = True
            state["snapped_best"]["upload_time"] = datetime.now().isoformat()
            print(f"  [best] Uploaded snapped checkpoint to GDrive!")
        else:
            print(f"  [best] Upload failed (checkpoint saved locally)")

    return True


def is_device_busy(device: str = "auto") -> bool:
    """Check if TPU/MPS is busy (other process using it).

    Note: CUDA/NVIDIA GPUs support concurrent access, so we don't check those.
    TPU and MPS are exclusive - only one heavy process at a time.

    Returns True if device appears busy, False if free.
    """
    try:
        # CUDA GPUs support concurrent access - no need to wait
        if device == "cuda" or device.startswith("cuda:"):
            return False

        # Check for TPU (XLA) - exclusive access, check /dev/vfio/0
        if device in ("auto", "tpu", "xla") or device.startswith("xla"):
            # Check if TPU device exists
            if not os.path.exists("/dev/vfio/0"):
                return False  # No TPU device, not busy

            # Try lsof first (check if device is open by any process)
            result = subprocess.run(
                ["lsof", "/dev/vfio/0"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                # Device is in use - check if it's our process
                current_pid = str(os.getpid())
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 2:
                        pid = parts[1]
                        if pid != current_pid:
                            return True  # Another process is using TPU
                return False  # Only our process (or no process)

            # Fallback: try fuser
            result = subprocess.run(
                ["fuser", "-v", "/dev/vfio/0"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                # fuser outputs PIDs to stderr with -v
                output = result.stderr.strip() if result.stderr else result.stdout.strip()
                if output:
                    current_pid = str(os.getpid())
                    # Parse fuser output (PIDs are space-separated)
                    pids = [p.strip() for p in output.split() if p.strip().isdigit()]
                    other_pids = [p for p in pids if p != current_pid]
                    if other_pids:
                        return True
            return False

        # MPS (Apple Silicon) - exclusive for heavy workloads
        if device in ("auto", "mps"):
            result = subprocess.run(
                ["pgrep", "-f", "python.*measure_perplexity"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                pids = [p.strip() for p in result.stdout.strip().split('\n') if p.strip()]
                current_pid = str(os.getpid())
                other_pids = [p for p in pids if p != current_pid]
                if other_pids:
                    return True
            return False

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass

    return False


def wait_for_device(device: str = "auto", check_interval: int = 60, max_wait: int = 3600) -> bool:
    """Wait until device is free.

    Args:
        device: Device type (auto, cuda, tpu, mps)
        check_interval: Seconds between checks (default: 60)
        max_wait: Maximum wait time in seconds (default: 3600 = 1 hour)

    Returns:
        True if device became free, False if max_wait exceeded
    """
    waited = 0
    while is_device_busy(device):
        if waited >= max_wait:
            print(f"  [warn] Max wait time ({max_wait}s) exceeded, proceeding anyway")
            return False
        print(f"  [wait] Device busy, waiting {check_interval}s... (total waited: {waited}s)")
        time.sleep(check_interval)
        waited += check_interval
    return True


def is_baked(state: dict, step: int) -> bool:
    """Check if step is already baked (legacy, kept for compatibility)."""
    e = state["entries"].get(str(step))
    return bool(e and e.get("bake_ok"))


def list_checkpoints_remote(remote_dir: Path, pattern: str) -> List[int]:
    """List checkpoint steps from remote directory."""
    pat = re.compile(pattern)
    steps = []
    if not remote_dir.exists():
        return steps
    for fn in os.listdir(remote_dir):
        m = pat.match(fn)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def list_checkpoints_local(local_dir: Path, pattern: str) -> List[int]:
    """List checkpoint steps from local directory."""
    pat = re.compile(pattern)
    steps = []
    if not local_dir.exists():
        return steps
    for fn in os.listdir(local_dir):
        m = pat.match(fn)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def get_checkpoint_name(step: int) -> str:
    """Get checkpoint filename for a step."""
    return f"checkpoint_step{step}.pt"


def get_baked_name(step: int) -> str:
    """Get baked filename for a step."""
    return f"baked_step{step}.pt"


def compute_md5(file_path: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def force_download_best(run_name: str, local_dir: Path, timeout: int = 600) -> Optional[Path]:
    """Force re-download best_state_dict.pt from Google Drive."""
    best_name = "best_state_dict.pt"
    local_best = local_dir / best_name

    # Remove existing file to force fresh download
    if local_best.exists():
        local_best.unlink()
        print(f"  [download] Removed existing {best_name}")

    print(f"  [download] Downloading {best_name}...")
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "gdrive_sync.py"),
        "down", run_name, "--only", best_name
    ]

    try:
        result = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [error] Download failed: {result.stderr[-500:] if result.stderr else 'unknown'}")
            return None
    except subprocess.TimeoutExpired:
        print(f"  [error] Download timed out")
        return None

    if not local_best.exists():
        print(f"  [error] Download claimed success but file missing")
        return None

    file_size = local_best.stat().st_size
    if file_size < 1000:  # Less than 1KB is definitely corrupt/empty
        print(f"  [error] Downloaded file is too small ({file_size} bytes) - likely missing on GDrive")
        local_best.unlink()  # Remove corrupt file
        return None

    print(f"  [download] Done: {file_size / 1e6:.1f} MB")
    return local_best


def process_best_checkpoint(
    run_name: str,
    local_dir: Path,
    state: dict,
    state_path: Path,
    config: str,
    dtype: str,
    device: str,
    max_chunks: Optional[int] = None,
    local_only: bool = False,
) -> bool:
    """Process best_state_dict.pt with md5-based deduplication.

    Returns True if processed successfully, False otherwise.
    """
    print("\n" + "=" * 70)
    print("PROCESSING: best_state_dict.pt")
    print("=" * 70)

    best_name = "best_state_dict.pt"
    local_best = local_dir / best_name

    # 1) Download (force re-download unless local-only)
    if not local_only:
        local_best = force_download_best(run_name, local_dir)
        if local_best is None:
            print("  [error] Failed to download best_state_dict.pt")
            return False
    else:
        if not local_best.exists():
            print(f"  [error] {best_name} not found locally")
            return False
        print(f"  [local] Using existing {best_name} ({local_best.stat().st_size / 1e6:.1f} MB)")

    # 2) Compute MD5
    print("  [md5] Computing hash...")
    md5_full = compute_md5(local_best)
    md5_short = md5_full[-5:]  # Last 5 characters
    print(f"  [md5] {md5_full} (short: {md5_short})")

    # 3) Check if this MD5 already processed
    best_entries = state.setdefault("best_entries", {})
    if md5_short in best_entries:
        entry = best_entries[md5_short]
        if entry.get("ppl_ok"):
            ppl = entry.get("ppl", "?")
            baked_name = entry.get("baked_name", "?")
            print(f"  [skip] MD5 {md5_short} already processed: PPL={ppl}, baked={baked_name}")
            return True

    # 4) Bake
    baked_tmp = local_dir / "baked_best.pt"
    baked_final = local_dir / f"baked_best_{md5_short}.pt"

    print(f"  [bake] {best_name} → baked_best_{md5_short}.pt...")
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "bake_lut.py"),
        str(local_best), str(baked_tmp)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [error] Bake failed:\n{result.stderr[-1000:] if result.stderr else 'unknown'}")
        return False

    if not baked_tmp.exists():
        print(f"  [error] Bake failed to create output")
        return False

    # Rename to include md5 suffix
    if baked_final.exists():
        baked_final.unlink()
    baked_tmp.rename(baked_final)
    print(f"  [bake] Done: {baked_final.name} ({baked_final.stat().st_size / 1e6:.1f} MB)")

    # Update state after bake
    best_entries[md5_short] = {
        "md5_full": md5_full,
        "md5_short": md5_short,
        "source": best_name,
        "baked_name": baked_final.name,
        "baked_path": str(baked_final),
        "bake_ok": True,
        "bake_time": datetime.now().isoformat(),
    }
    atomic_write_json(state_path, state)

    # 5) Wait for device and run perplexity
    wait_for_device(device)

    print(f"  [ppl] Running perplexity on {baked_final.name}...")
    try:
        ppl, xe, tokens, elapsed, _ = run_ppl(
            baked_final, config, dtype, device, max_chunks
        )
    except Exception as e:
        print(f"  [error] Perplexity failed: {e}")
        return False

    # Update state after PPL
    best_entries[md5_short].update({
        "ppl_ok": True,
        "ppl": ppl,
        "xe": xe,
        "tokens": tokens,
        "ppl_seconds": round(elapsed, 1),
        "ppl_time": datetime.now().isoformat(),
    })
    atomic_write_json(state_path, state)

    print(f"  [done] PPL={ppl:.2f}, XE={xe:.4f}, time={elapsed:.1f}s")
    print(f"  [done] Saved as: {baked_final.name}")

    # Check if this is the new best PPL and snap/upload if so
    check_and_snap_best(
        state=state,
        current_ppl=ppl,
        current_source=f"best_{md5_short}",
        baked_path=baked_final,
        local_dir=local_dir,
        run_name=run_name,
        config=config,
        upload=not local_only,
    )
    atomic_write_json(state_path, state)

    # 6) Print summary of all best entries
    print("\n" + "-" * 50)
    print("BEST CHECKPOINTS SUMMARY")
    print("-" * 50)
    print(f"{'MD5':>8} | {'PPL':>10} | {'Baked File'}")
    print("-" * 50)
    for md5_key, entry in sorted(best_entries.items(), key=lambda x: x[1].get("ppl", 999)):
        ppl_val = entry.get("ppl")
        ppl_str = f"{ppl_val:.2f}" if ppl_val else "-"
        baked = entry.get("baked_name", "-")
        print(f"{md5_key:>8} | {ppl_str:>10} | {baked}")
    print("-" * 50)

    return True


def ensure_download(run_name: str, local_dir: Path, step: int, timeout: int = 600) -> Path:
    """Download checkpoint if not present (blocking)."""
    ckpt_name = get_checkpoint_name(step)
    ckpt = local_dir / ckpt_name

    if ckpt.exists() and ckpt.stat().st_size > 0:
        print(f"  [local] {ckpt_name} already exists ({ckpt.stat().st_size / 1e6:.1f} MB)")
        return ckpt

    print(f"  [download] {ckpt_name}...")
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "gdrive_sync.py"),
        "down", run_name, "--only", ckpt_name
    ]

    try:
        result = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [error] Download failed: {result.stderr[-500:]}")
            raise RuntimeError(f"Download failed for {ckpt_name}")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Download timed out for {ckpt_name}")

    if not ckpt.exists():
        raise RuntimeError(f"Download claimed success but file missing: {ckpt}")

    print(f"  [download] Done: {ckpt.stat().st_size / 1e6:.1f} MB")
    return ckpt


def start_prefetch(run_name: str, local_dir: Path, step: int) -> Optional[subprocess.Popen]:
    """Start background download of checkpoint (non-blocking)."""
    ckpt_name = get_checkpoint_name(step)
    ckpt = local_dir / ckpt_name

    if ckpt.exists() and ckpt.stat().st_size > 0:
        return None

    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "gdrive_sync.py"),
        "down", run_name, "--only", ckpt_name
    ]
    # Redirect output to devnull for background process
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def bake(local_dir: Path, step: int) -> Path:
    """Bake LUT for checkpoint."""
    in_ckpt = local_dir / get_checkpoint_name(step)
    out_baked = local_dir / get_baked_name(step)

    if out_baked.exists() and out_baked.stat().st_size > 0:
        print(f"  [bake] {out_baked.name} already exists")
        return out_baked

    print(f"  [bake] {in_ckpt.name} → {out_baked.name}...")
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "bake_lut.py"),
        str(in_ckpt), str(out_baked)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [error] Bake failed:\n{result.stderr[-1000:]}")
        raise RuntimeError(f"Bake failed for step {step}")

    if not out_baked.exists():
        raise RuntimeError(f"Bake failed to create: {out_baked}")

    print(f"  [bake] Done: {out_baked.stat().st_size / 1e6:.1f} MB")
    return out_baked


def run_ppl(
    baked_path: Path,
    config: str,
    dtype: str,
    device: str = "auto",
    max_chunks: Optional[int] = None,
) -> Tuple[float, float, int, float, str]:
    """Run perplexity measurement. Returns (ppl, xe, tokens, seconds, raw_output)."""
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "measure_perplexity.py"),
        str(baked_path),
        "--config", config,
        "--dtype", dtype,
        "--device", device,
    ]
    if max_chunks:
        cmd.extend(["--max-chunks", str(max_chunks)])

    print(f"  [ppl] Running perplexity on {baked_path.name}...")
    start = time.time()

    # Combine stdout and stderr, and stream output to console
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Combine stderr into stdout
        text=True,
    )
    elapsed = time.time() - start

    out = result.stdout

    # Print output for visibility (in case of issues)
    print(out)

    if result.returncode != 0:
        print(f"  [error] Perplexity failed (rc={result.returncode})")
        raise RuntimeError(f"Perplexity failed for {baked_path}")

    # Parse results using regex for robustness
    ppl = None
    xe = None
    tokens = None

    # Match "Perplexity:" followed by whitespace and a number
    ppl_match = re.search(r'Perplexity:\s+([\d.]+)', out)
    if ppl_match:
        ppl = float(ppl_match.group(1))

    # Match "Cross-entropy:" followed by whitespace and a number
    xe_match = re.search(r'Cross-entropy:\s+([\d.]+)', out)
    if xe_match:
        xe = float(xe_match.group(1))

    # Match "Tokens:" followed by whitespace and a number (with optional commas)
    tokens_match = re.search(r'Tokens:\s+([\d,]+)', out)
    if tokens_match:
        tokens = int(tokens_match.group(1).replace(",", ""))

    if ppl is None:
        print(f"  [warn] Could not parse perplexity from output")
        raise RuntimeError("Could not parse perplexity from output")

    return ppl, xe or 0.0, tokens or 0, elapsed, out


def print_summary(state: dict):
    """Print summary of all processed checkpoints."""
    entries = state.get("entries", {})
    best_entries = state.get("best_entries", {})

    if not entries and not best_entries:
        print("\nNo checkpoints processed yet.")
        return

    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"{'Step':>8} | {'PPL':>10} | {'XE':>10} | {'Tokens':>12} | {'Status'}")
    print("-" * 70)

    # Sort by step
    sorted_entries = sorted(entries.items(), key=lambda x: int(x[0]))

    best_ppl = float('inf')
    best_step = None

    for step_str, e in sorted_entries:
        step = int(step_str)
        ppl = e.get("ppl")
        xe = e.get("xe")
        tokens = e.get("tokens", 0)
        ppl_ok = e.get("ppl_ok", False)
        bake_ok = e.get("bake_ok", False)

        if ppl_ok and ppl is not None:
            status = "OK"
            if ppl < best_ppl:
                best_ppl = ppl
                best_step = step
        elif bake_ok:
            status = "baked"
        else:
            status = "pending"

        ppl_str = f"{ppl:.2f}" if ppl is not None else "-"
        xe_str = f"{xe:.4f}" if xe is not None else "-"
        tokens_str = f"{tokens:,}" if tokens else "-"

        print(f"{step:>8} | {ppl_str:>10} | {xe_str:>10} | {tokens_str:>12} | {status}")

    # Add best_state_dict entries (sorted by PPL)
    if best_entries:
        print("-" * 70)
        for md5_key, e in sorted(best_entries.items(), key=lambda x: x[1].get("ppl", 999)):
            ppl = e.get("ppl")
            xe = e.get("xe")
            tokens = e.get("tokens", 0)
            ppl_ok = e.get("ppl_ok", False)

            status = f"best_{md5_key}" if ppl_ok else f"best_{md5_key}(pending)"
            ppl_str = f"{ppl:.2f}" if ppl is not None else "-"
            xe_str = f"{xe:.4f}" if xe is not None else "-"
            tokens_str = f"{tokens:,}" if tokens else "-"

            print(f"{'best':>8} | {ppl_str:>10} | {xe_str:>10} | {tokens_str:>12} | {status}")

            # Track if any best entry beats step-based best
            if ppl_ok and ppl is not None and ppl < best_ppl:
                best_ppl = ppl
                best_step = f"best_{md5_key}"

    print("-" * 70)
    if best_step is not None:
        print(f"Best: {best_step} with PPL = {best_ppl:.2f}")

    # Show snapped best info if available
    snapped_best = state.get("snapped_best")
    if snapped_best:
        print(f"\nSnapped Best Checkpoint:")
        print(f"  Source:  {snapped_best.get('source', '?')}")
        print(f"  PPL:     {snapped_best.get('ppl', '?'):.2f}" if snapped_best.get('ppl') else "  PPL:     ?")
        print(f"  File:    {snapped_best.get('snapped_name', '?')}")
        uploaded = snapped_best.get('uploaded', False)
        print(f"  GDrive:  {'uploaded' if uploaded else 'not uploaded'}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline: Download → Bake → Perplexity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("run_name", nargs="?", default=None,
                        help="Run name (folder name in qwen3_runs)")
    parser.add_argument("-b", "--base-dir", metavar="FOLDER", default=None,
                        help="Base folder (alternative to run_name positional)")
    parser.add_argument("--drive-root", default=None,
                        help="Google Drive root (auto-detect if not specified)")
    parser.add_argument("--local-root", default="runs",
                        help="Local runs directory (default: runs)")
    parser.add_argument("--config", default="q4_r32",
                        help="Quantization config preset (default: q4_r32)")
    parser.add_argument("--dtype", default="fp16",
                        help="Model dtype (default: fp16)")
    parser.add_argument("--device", default="auto",
                        help="Device (auto, mps, cuda, cpu)")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Max checkpoints to process (0=all)")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="Max chunks for PPL (faster screening)")
    parser.add_argument("--pattern", default=r"checkpoint_step(\d+)\.pt$",
                        help="Checkpoint filename pattern with step group")
    parser.add_argument("--local-only", action="store_true",
                        help="Skip download, use local checkpoints only")
    parser.add_argument("--no-prefetch", action="store_true",
                        help="Disable background prefetch")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary from state file and exit")
    parser.add_argument("--clean-checkpoints", action="store_true",
                        help="Delete raw checkpoints after successful bake+ppl")
    parser.add_argument("--best", action="store_true",
                        help="Process best_state_dict.pt (force download, md5 dedup, bake, ppl)")
    parser.add_argument("--list", action="store_true",
                        help="List all checkpoint records (steps and best entries)")
    parser.add_argument("--delete", type=str, metavar="ENTRY",
                        help="Delete a record by step number (e.g., '100') or best md5 (e.g., 'best_abc12')")

    args = parser.parse_args()

    # Determine run name from positional arg or -b option
    run_name = args.run_name or args.base_dir
    if not run_name:
        parser.print_help()
        print("\nError: run_name or -b/--base-dir is required")
        return 1

    # Strip runs/ prefix if present (like gdrive_sync.py)
    if run_name.startswith("runs/"):
        run_name = run_name[5:]
    if run_name.startswith("runs\\"):
        run_name = run_name[5:]

    # Setup paths
    local_dir = Path(args.local_root) / run_name
    local_dir.mkdir(parents=True, exist_ok=True)

    # Try to sync ppl_state.json from Google Drive first (get latest results)
    if not args.local_only:
        print(f"[sync] Checking for ppl_state.json on Google Drive...")
        if sync_state_from_gdrive(run_name):
            print(f"[sync] Downloaded ppl_state.json from Google Drive")
        else:
            print(f"[sync] No ppl_state.json on Google Drive (or sync failed)")

    state_path = local_dir / "ppl_state.json"
    state = load_state(state_path)
    state["run"] = run_name
    state["config"] = args.config
    state["dtype"] = args.dtype

    # Handle --list: show all records with details
    if args.list:
        entries = state.get("entries", {})
        best_entries = state.get("best_entries", {})
        snapped_best = state.get("snapped_best")

        print("\n" + "=" * 70)
        print(f"CHECKPOINT RECORDS FOR: {run_name}")
        print("=" * 70)

        if entries:
            print(f"\nStep-based entries ({len(entries)} total):")
            print("-" * 70)
            print(f"{'Step':>8} | {'PPL':>10} | {'Baked File':30} | {'Delete Key'}")
            print("-" * 70)
            for step_str in sorted(entries.keys(), key=int):
                e = entries[step_str]
                ppl = e.get("ppl")
                ppl_str = f"{ppl:.2f}" if ppl is not None else "-"
                baked = Path(e.get("baked_path", "")).name if e.get("baked_path") else "-"
                print(f"{step_str:>8} | {ppl_str:>10} | {baked:30} | {step_str}")
        else:
            print("\nNo step-based entries.")

        if best_entries:
            print(f"\nBest checkpoint entries ({len(best_entries)} total):")
            print("-" * 70)
            print(f"{'MD5':>8} | {'PPL':>10} | {'Baked File':30} | {'Delete Key'}")
            print("-" * 70)
            for md5_key, e in sorted(best_entries.items(), key=lambda x: x[1].get("ppl", 999)):
                ppl = e.get("ppl")
                ppl_str = f"{ppl:.2f}" if ppl is not None else "-"
                baked = e.get("baked_name", "-")
                print(f"{md5_key:>8} | {ppl_str:>10} | {baked:30} | best_{md5_key}")
        else:
            print("\nNo best checkpoint entries.")

        if snapped_best:
            print(f"\nSnapped best checkpoint:")
            print(f"  Source: {snapped_best.get('source', '?')}")
            print(f"  PPL:    {snapped_best.get('ppl', '?')}")
            print(f"  File:   {snapped_best.get('snapped_name', '?')}")
            print(f"  Delete: snapped_best")

        print("\n" + "=" * 70)
        print("To delete a record:")
        print("  python scripts/pipeline_bake_ppl.py <run> --delete <key>")
        print("  e.g., --delete 100        (delete step 100)")
        print("  e.g., --delete best_abc12 (delete best entry)")
        print("  e.g., --delete snapped_best (delete snapped best info)")
        print("=" * 70)
        return 0

    # Handle --delete: remove a specific record
    if args.delete:
        key = args.delete
        entries = state.get("entries", {})
        best_entries = state.get("best_entries", {})
        deleted = False

        # Check if it's a step number
        if key.isdigit():
            if key in entries:
                entry = entries[key]
                ppl = entry.get("ppl", "?")
                baked_path = entry.get("baked_path")
                del entries[key]
                deleted = True
                print(f"Deleted step {key} (PPL={ppl})")
                if baked_path and Path(baked_path).exists():
                    print(f"  Note: Baked file still exists: {baked_path}")
                    print(f"  To delete file: rm '{baked_path}'")
            else:
                print(f"Error: Step {key} not found in records")
                return 1

        # Check if it's a best entry (best_XXXXX)
        elif key.startswith("best_"):
            md5_key = key[5:]  # Remove "best_" prefix
            if md5_key in best_entries:
                entry = best_entries[md5_key]
                ppl = entry.get("ppl", "?")
                baked_path = entry.get("baked_path")
                del best_entries[md5_key]
                deleted = True
                print(f"Deleted best entry {md5_key} (PPL={ppl})")
                if baked_path and Path(baked_path).exists():
                    print(f"  Note: Baked file still exists: {baked_path}")
                    print(f"  To delete file: rm '{baked_path}'")
            else:
                print(f"Error: Best entry {md5_key} not found in records")
                return 1

        # Check if it's snapped_best
        elif key == "snapped_best":
            if "snapped_best" in state:
                snapped = state["snapped_best"]
                snapped_path = snapped.get("snapped_path")
                del state["snapped_best"]
                deleted = True
                print(f"Deleted snapped_best info")
                if snapped_path and Path(snapped_path).exists():
                    print(f"  Note: Snapped file still exists: {snapped_path}")
                    print(f"  To delete file: rm '{snapped_path}'")
            else:
                print(f"Error: No snapped_best info in records")
                return 1
        else:
            print(f"Error: Unknown key format '{key}'")
            print("  Use step number (e.g., '100'), 'best_XXXXX', or 'snapped_best'")
            return 1

        if deleted:
            atomic_write_json(state_path, state)
            print(f"State saved to: {state_path}")

            # Sync to GDrive
            if not args.local_only:
                print("Syncing updated state to Google Drive...")
                if sync_state_to_gdrive(run_name):
                    print("Done")

        return 0

    # Handle --summary
    if args.summary:
        print_summary(state)
        return 0

    # Handle --best: process best_state_dict.pt and exit
    if args.best:
        success = process_best_checkpoint(
            run_name=run_name,
            local_dir=local_dir,
            state=state,
            state_path=state_path,
            config=args.config,
            dtype=args.dtype,
            device=args.device,
            max_chunks=args.max_chunks,
            local_only=args.local_only,
        )
        # Sync state to Google Drive
        if not args.local_only:
            print("\n[sync] Syncing ppl_state.json to Google Drive...")
            if sync_state_to_gdrive(run_name):
                print("[sync] Done")
        return 0 if success else 1

    # Determine drive root
    if args.drive_root:
        drive_root = Path(args.drive_root)
    else:
        gdrive = find_gdrive_root()
        if gdrive:
            drive_root = gdrive / "qwen3_runs"
        else:
            drive_root = Path("/content/drive/MyDrive/qwen3_runs")

    remote_dir = drive_root / run_name

    print("=" * 70)
    print("PIPELINE: BAKE + PERPLEXITY")
    print("=" * 70)
    print(f"Run name:    {run_name}")
    print(f"Remote dir:  {remote_dir}")
    print(f"Local dir:   {local_dir}")
    print(f"Config:      {args.config}")
    print(f"Dtype:       {args.dtype}")
    print(f"Device:      {args.device}")
    print(f"State file:  {state_path}")
    print("=" * 70)

    # Discover checkpoints
    if args.local_only:
        steps = list_checkpoints_local(local_dir, args.pattern)
        print(f"\nLocal mode: found {len(steps)} checkpoints")
    else:
        if not remote_dir.exists():
            print(f"\nError: Remote directory not found: {remote_dir}")
            print("Check that Google Drive is mounted and run name is correct.")
            return 1
        steps = list_checkpoints_remote(remote_dir, args.pattern)
        print(f"\nFound {len(steps)} checkpoints in remote directory")

    if not steps:
        print("No checkpoints found matching pattern.")
        return 1

    if args.max_steps and args.max_steps > 0:
        steps = steps[:args.max_steps]
        print(f"Limited to first {args.max_steps} checkpoints")

    print(f"Steps to process: {steps[:10]}{'...' if len(steps) > 10 else ''}")
    print()

    # Load perplexity cache (results/perplexity.json)
    ppl_cache = load_ppl_cache()
    if ppl_cache:
        print(f"[cache] Loaded {len(ppl_cache)} entries from results/perplexity.json")

    # Wait for device before starting pipeline (TPU is exclusive)
    print(f"\n[device] Checking if {args.device} is available...")
    if is_device_busy(args.device):
        print(f"[device] Device busy, waiting for it to be free...")
        wait_for_device(args.device)
        print(f"[device] Device is now free, starting pipeline")
    else:
        print(f"[device] Device is free")

    # Pipeline loop
    prefetch_proc = None
    processed = 0
    skipped = 0
    cached = 0

    for i, step in enumerate(steps):
        print(f"\n[{i+1}/{len(steps)}] Step {step}")
        print("-" * 40)

        if is_done(state, step):
            ppl = state["entries"][str(step)].get("ppl", "?")
            print(f"  [skip] Already done (PPL={ppl})")
            skipped += 1
            continue

        try:
            # 1) Ensure current checkpoint exists locally
            if not args.local_only:
                ensure_download(run_name, local_dir, step)

            ckpt_path = local_dir / get_checkpoint_name(step)
            if not ckpt_path.exists():
                print(f"  [error] Checkpoint not found: {ckpt_path}")
                continue

            # 2) Start prefetch of next checkpoint
            if not args.no_prefetch and not args.local_only and prefetch_proc is None:
                if i + 1 < len(steps):
                    nxt = steps[i + 1]
                    if not is_done(state, nxt):
                        nxt_ckpt = local_dir / get_checkpoint_name(nxt)
                        if not nxt_ckpt.exists():
                            print(f"  [prefetch] Starting download of step {nxt}")
                            prefetch_proc = start_prefetch(run_name, local_dir, nxt)

            # 3) Bake
            baked_path = bake(local_dir, step)

            # Update state after bake
            state["entries"].setdefault(str(step), {})
            state["entries"][str(step)].update({
                "step": step,
                "ckpt_name": get_checkpoint_name(step),
                "local_ckpt_path": str(ckpt_path),
                "baked_path": str(baked_path),
                "bake_ok": True,
                "bake_time": datetime.now().isoformat(),
            })
            atomic_write_json(state_path, state)

            # 4) Perplexity - check cache first (per-run state has priority)
            cached_result = get_cached_ppl(run_name, step, ppl_cache, state)
            if cached_result:
                ppl = cached_result["ppl"]
                xe = cached_result["xe"]
                tokens = cached_result["tokens"]
                elapsed = 0.0
                print(f"  [cache] Found in results/perplexity.json: PPL={ppl:.2f}")
                cached += 1
            else:
                # Wait for device to be free before running PPL
                wait_for_device(args.device)
                ppl, xe, tokens, elapsed, _ = run_ppl(
                    baked_path, args.config, args.dtype, args.device, args.max_chunks
                )
                processed += 1

            # Update state after PPL
            state["entries"][str(step)].update({
                "ppl_ok": True,
                "ppl": ppl,
                "xe": xe,
                "tokens": tokens,
                "ppl_seconds": round(elapsed, 1),
                "ppl_time": datetime.now().isoformat(),
            })
            atomic_write_json(state_path, state)

            print(f"  [done] PPL={ppl:.2f}, XE={xe:.4f}, time={elapsed:.1f}s")

            # Check if this is the new best PPL and snap/upload if so
            check_and_snap_best(
                state=state,
                current_ppl=ppl,
                current_source=f"step_{step}",
                baked_path=baked_path,
                local_dir=local_dir,
                run_name=run_name,
                config=args.config,
                upload=not args.local_only,
            )
            atomic_write_json(state_path, state)

            # Sync state to Google Drive (backup results)
            if not args.local_only:
                sync_state_to_gdrive(run_name)

            # 5) Optionally clean up raw checkpoint
            if args.clean_checkpoints and ckpt_path.exists():
                ckpt_path.unlink()
                print(f"  [clean] Deleted {ckpt_path.name}")

            # 6) Check prefetch status
            if prefetch_proc is not None and prefetch_proc.poll() is not None:
                rc = prefetch_proc.returncode
                if rc != 0:
                    print(f"  [warn] Prefetch failed (rc={rc})")
                prefetch_proc = None

        except Exception as e:
            print(f"  [error] {e}")
            # Continue to next step
            continue

    # Wait for final prefetch
    if prefetch_proc is not None:
        print("\nWaiting for background prefetch to complete...")
        prefetch_proc.wait()

    # Final sync to Google Drive
    if not args.local_only:
        print("\n[sync] Final sync of ppl_state.json to Google Drive...")
        if sync_state_to_gdrive(run_name):
            print("[sync] Done")

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"COMPLETE: processed={processed}, cached={cached}, skipped={skipped}")
    print(f"State saved to: {state_path}")

    print_summary(state)

    return 0


if __name__ == "__main__":
    sys.exit(main())
