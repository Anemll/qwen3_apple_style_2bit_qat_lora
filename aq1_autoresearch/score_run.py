#!/usr/bin/env python3
"""Summarize AQ1 candidate artifacts into a stable score record."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score an AQ1 candidate directory")
    parser.add_argument("--run-dir", required=True, help="Candidate directory")
    parser.add_argument("--checkpoint", default=None, help="Preferred checkpoint path to score")
    parser.add_argument("--ppl-log", default=None, help="Optional perplexity log path")
    parser.add_argument("--results-json", default=None, help="Optional results/perplexity.json path")
    parser.add_argument("--baseline-run-dir", default=None, help="Optional baseline run dir for size budget comparison")
    parser.add_argument("--max-size-growth-pct", type=float, default=None, help="Optional maximum projected payload growth versus baseline")
    parser.add_argument(
        "--float-storage-bits",
        type=int,
        default=16,
        help="Bits to assume for scale/LUT storage in projected deployment size (default: 16)",
    )
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    return parser.parse_args()


def maybe_float(value: str | None) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    return float(value.replace(",", ""))


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def pick_saved_fields(saved: dict[str, Any] | None, keys: list[str]) -> dict[str, Any]:
    if not saved:
        return {}
    return {key: saved.get(key) for key in keys if key in saved}


def maybe_import_torch() -> Any | None:
    try:
        import torch  # type: ignore
    except Exception:
        return None
    return torch


def load_checkpoint_state_dict(checkpoint: Path) -> tuple[dict[str, Any] | None, str | None]:
    torch = maybe_import_torch()
    if torch is None:
        return None, "torch unavailable; run score_run.py inside the AQ1 venv to estimate size"

    try:
        obj = torch.load(checkpoint, map_location="cpu")
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

    if isinstance(obj, dict) and isinstance(obj.get("model_state_dict"), dict):
        state_dict = obj["model_state_dict"]
    elif isinstance(obj, dict):
        state_dict = obj
    else:
        return None, f"unsupported checkpoint object: {type(obj).__name__}"

    return state_dict, None


def estimate_quantized_payload(checkpoint: Path | None, float_storage_bits: int = 16) -> dict[str, Any]:
    if checkpoint is None:
        return {}
    if not checkpoint.exists():
        return {"size_estimate_error": f"missing checkpoint: {checkpoint}"}
    if float_storage_bits <= 0:
        return {"size_estimate_error": f"invalid float_storage_bits: {float_storage_bits}"}

    state_dict, error = load_checkpoint_state_dict(checkpoint)
    if error is not None or state_dict is None:
        return {"size_estimate_error": error}

    required_suffixes = ("lut", "scale_A", "scale_B", "rank_magnitude")
    layer_prefixes: dict[str, set[str]] = {}
    for key, value in state_dict.items():
        if not hasattr(value, "numel"):
            continue
        suffix = key.rsplit(".", 1)[-1]
        if suffix in required_suffixes:
            prefix = key[: -(len(suffix) + 1)]
            layer_prefixes.setdefault(prefix, set()).add(suffix)

    total_index_bits = 0
    total_scale_bits = 0
    total_lut_bits = 0
    total_weight_count = 0
    quant_layer_count = 0
    family_weight_count = {"attn": 0, "mlp": 0, "other": 0}
    family_payload_bits = {"attn": 0, "mlp": 0, "other": 0}

    for prefix, suffixes in sorted(layer_prefixes.items()):
        if not all(suffix in suffixes for suffix in required_suffixes):
            continue

        lut = state_dict[f"{prefix}.lut"]
        scale_a = state_dict[f"{prefix}.scale_A"]
        scale_b = state_dict[f"{prefix}.scale_B"]
        rank_magnitude = state_dict[f"{prefix}.rank_magnitude"]

        weight_tensor = state_dict.get(f"{prefix}._indices")
        if weight_tensor is None:
            weight_tensor = state_dict.get(f"{prefix}._Q")
        if weight_tensor is not None and hasattr(weight_tensor, "numel"):
            weight_count = int(weight_tensor.numel())
        else:
            weight_count = int(scale_a.shape[0] * scale_b.shape[1])

        lut_size = int(lut.numel())
        lut_bits = int(math.ceil(math.log2(lut_size))) if lut_size > 1 else 0
        index_bits = weight_count * lut_bits
        scale_bits = (int(scale_a.numel()) + int(scale_b.numel()) + int(rank_magnitude.numel())) * float_storage_bits
        lut_bits_total = lut_size * float_storage_bits
        payload_bits = index_bits + scale_bits + lut_bits_total

        if ".self_attn." in prefix:
            family = "attn"
        elif ".mlp." in prefix:
            family = "mlp"
        else:
            family = "other"

        total_index_bits += index_bits
        total_scale_bits += scale_bits
        total_lut_bits += lut_bits_total
        total_weight_count += weight_count
        quant_layer_count += 1
        family_weight_count[family] += weight_count
        family_payload_bits[family] += payload_bits

    projected_payload_bits = total_index_bits + total_scale_bits + total_lut_bits
    projected_payload_bytes = math.ceil(projected_payload_bits / 8) if projected_payload_bits else None
    projected_payload_mib = (
        projected_payload_bits / 8 / 1024 / 1024 if projected_payload_bits else None
    )
    avg_bits_per_weight = (
        projected_payload_bits / total_weight_count if total_weight_count else None
    )

    family_avg_bits = {}
    for family, weight_count in family_weight_count.items():
        if weight_count:
            family_avg_bits[family] = family_payload_bits[family] / weight_count

    return {
        "size_estimate_mode": "packed_lut_indices_plus_fp16_scales",
        "size_estimate_error": None,
        "float_storage_bits": float_storage_bits,
        "quant_layer_count": quant_layer_count,
        "quant_weight_count": total_weight_count,
        "projected_index_bits": total_index_bits,
        "projected_scale_bits": total_scale_bits,
        "projected_lut_bits": total_lut_bits,
        "projected_payload_bits": projected_payload_bits,
        "projected_payload_bytes": projected_payload_bytes,
        "projected_payload_mib": projected_payload_mib,
        "avg_bits_per_weight": avg_bits_per_weight,
        "family_avg_bits_per_weight": family_avg_bits,
    }


def find_run_candidate_checkpoint(run_dir: Path, requested_checkpoint: str | None = None) -> Path | None:
    checkpoints = find_candidate_checkpoints(run_dir)
    if requested_checkpoint:
        requested = Path(requested_checkpoint).expanduser()
        if not requested.is_absolute():
            requested = (run_dir / requested).resolve()
        else:
            requested = requested.resolve()
        return requested
    return checkpoints[0] if checkpoints else None


def compare_size_to_baseline(
    candidate_checkpoint: Path | None,
    candidate_size: dict[str, Any] | None,
    baseline_run_dir: str | None,
    max_size_growth_pct: float | None,
    float_storage_bits: int,
) -> dict[str, Any]:
    if candidate_checkpoint is None or baseline_run_dir is None:
        return {}

    baseline_dir = Path(baseline_run_dir).expanduser().resolve()
    if not baseline_dir.exists():
        return {"size_budget_error": f"missing baseline run dir: {baseline_dir}"}

    baseline_checkpoint = find_run_candidate_checkpoint(baseline_dir)
    if baseline_checkpoint is None:
        return {"size_budget_error": f"no candidate checkpoint found in baseline run dir: {baseline_dir}"}

    baseline_size = (
        candidate_size
        if baseline_checkpoint.resolve() == candidate_checkpoint.resolve() and candidate_size is not None
        else estimate_quantized_payload(baseline_checkpoint, float_storage_bits=float_storage_bits)
    )
    if baseline_size.get("projected_payload_bits") is None:
        return {
            "size_budget_error": baseline_size.get("size_estimate_error")
            or f"failed to estimate baseline size from {baseline_checkpoint}"
        }

    candidate_size = candidate_size or estimate_quantized_payload(candidate_checkpoint, float_storage_bits=float_storage_bits)
    candidate_bits = candidate_size.get("projected_payload_bits")
    baseline_bits = baseline_size.get("projected_payload_bits")
    if candidate_bits is None or baseline_bits is None:
        return {
            "size_budget_error": candidate_size.get("size_estimate_error")
            or f"failed to estimate candidate size from {candidate_checkpoint}"
        }

    ratio = candidate_bits / baseline_bits if baseline_bits else None
    growth_pct = ((ratio - 1.0) * 100.0) if ratio is not None else None

    budget_payload_bits = None
    budget_payload_mib = None
    budget_avg_bits = None
    size_ok = None
    if max_size_growth_pct is not None:
        budget_payload_bits = baseline_bits * (1.0 + max_size_growth_pct / 100.0)
        budget_payload_mib = budget_payload_bits / 8 / 1024 / 1024
        if baseline_size.get("avg_bits_per_weight") is not None:
            budget_avg_bits = baseline_size["avg_bits_per_weight"] * (1.0 + max_size_growth_pct / 100.0)
        size_ok = candidate_bits <= budget_payload_bits + 1e-9

    return {
        "baseline_checkpoint": str(baseline_checkpoint),
        "baseline_projected_payload_mib": baseline_size.get("projected_payload_mib"),
        "baseline_avg_bits_per_weight": baseline_size.get("avg_bits_per_weight"),
        "size_ratio_vs_baseline": ratio,
        "size_growth_pct_vs_baseline": growth_pct,
        "size_budget_growth_pct": max_size_growth_pct,
        "size_budget_payload_mib": budget_payload_mib,
        "size_budget_avg_bits_per_weight": budget_avg_bits,
        "size_ok": size_ok,
        "size_budget_error": None,
    }


def load_loss_csv(loss_csv: Path) -> tuple[list[dict[str, float | None]], float | None]:
    rows: list[dict[str, float | None]] = []
    last_elapsed = None
    with open(loss_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_loss = maybe_float(row.get("train_loss"))
            eval_loss = maybe_float(row.get("eval_loss"))
            elapsed = maybe_float(row.get("elapsed_sec"))
            rows.append(
                {
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                    "elapsed_sec": elapsed,
                }
            )
            if elapsed is not None:
                last_elapsed = elapsed
    return rows, last_elapsed


def extract_init_quick_ppl(metrics: dict[str, Any]) -> dict[str, float | None]:
    ppl = (((metrics.get("steps") or {}).get("perplexity") or {}).get("perplexity") or {})
    return {
        "quick_perplexity": ppl.get("perplexity"),
        "quick_cross_entropy": ppl.get("cross_entropy"),
        "quick_tokens": ppl.get("tokens"),
        "quick_time_sec": ppl.get("time"),
    }


def find_candidate_checkpoints(run_dir: Path) -> list[Path]:
    preferred = [
        "v2_tightened.pt",
        "tightQ_all.pt",
        "hybrid.pt",
        "snapped_fp16.pt",
        "v2_initial.pt",
        "best_state_dict.pt",
        "final_state_dict.pt",
    ]
    found: list[Path] = []
    seen: set[Path] = set()

    for name in preferred:
        path = run_dir / name
        if path.exists() and path not in seen:
            found.append(path)
            seen.add(path)

    for path in sorted(run_dir.glob("*.pt")):
        if path not in seen:
            found.append(path)
            seen.add(path)

    return found


def normalize_ppl_key(checkpoint: Path) -> list[str]:
    keys = [str(checkpoint)]

    try:
        rel_repo = checkpoint.relative_to(REPO_ROOT)
        keys.append(str(rel_repo))
    except ValueError:
        pass

    runs_root = REPO_ROOT / "runs"
    try:
        rel_runs = checkpoint.relative_to(runs_root)
        keys.append(str(rel_runs))
    except ValueError:
        pass

    deduped: list[str] = []
    seen = set()
    for key in keys:
        if key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


def parse_ppl_log(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")

    def match_float(pattern: str) -> float | None:
        match = re.search(pattern, text, re.MULTILINE)
        return float(match.group(1).replace(",", "")) if match else None

    def match_text(pattern: str) -> str | None:
        match = re.search(pattern, text, re.MULTILINE)
        return match.group(1).strip() if match else None

    return {
        "source": str(log_path),
        "key": match_text(r"^Key:\s*(.+)$"),
        "perplexity": match_float(r"^Perplexity:\s*([0-9][0-9,]*(?:\.[0-9]+)?)"),
        "cross_entropy": match_float(r"^Cross-entropy:\s*([0-9][0-9,]*(?:\.[0-9]+)?)"),
        "tokens": match_float(r"^Tokens:\s*([0-9][0-9,]*)"),
        "time_seconds": match_float(r"^Time:\s*([0-9][0-9,]*(?:\.[0-9]+)?)s"),
    }


def parse_init_log_quick_ppl(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")

    def match_float(pattern: str) -> float | None:
        match = re.search(pattern, text, re.MULTILINE)
        return float(match.group(1).replace(",", "")) if match else None

    def find_all_floats(pattern: str) -> list[float]:
        return [float(value.replace(",", "")) for value in re.findall(pattern, text, re.MULTILINE)]

    total_time_matches = find_all_floats(r"^\s*Total time:\s*([0-9][0-9,]*(?:\.[0-9]+)?)s")

    return {
        "source": str(log_path),
        "quick_perplexity": match_float(r"^\s*Quick PPL:\s*([0-9][0-9,]*(?:\.[0-9]+)?)"),
        "quick_cross_entropy": match_float(r"^\s*Cross-entropy:\s*([0-9][0-9,]*(?:\.[0-9]+)?) nats"),
        "quick_time_sec": match_float(r"^\s*Total time:\s*([0-9][0-9,]*(?:\.[0-9]+)?)s"),
        "elapsed_sec": total_time_matches[-1] if total_time_matches else None,
    }


def load_full_ppl(run_dir: Path, checkpoint: Path | None, ppl_log_arg: str | None, results_json_arg: str | None) -> dict[str, Any]:
    candidate_logs: list[Path] = []
    if ppl_log_arg:
        candidate_logs.append(Path(ppl_log_arg).expanduser().resolve())
    candidate_logs.extend(
        [
            run_dir / "perplexity.log",
            run_dir / "ppl.log",
            run_dir / "measure_perplexity.log",
        ]
    )

    for log_path in candidate_logs:
        if log_path.exists():
            parsed = parse_ppl_log(log_path)
            if parsed.get("perplexity") is not None:
                parsed["kind"] = "log"
                return parsed

    results_json = Path(results_json_arg).expanduser().resolve() if results_json_arg else REPO_ROOT / "results" / "perplexity.json"
    results = load_json(results_json)
    if not results or checkpoint is None:
        return {}

    for key in normalize_ppl_key(checkpoint):
        if key in results:
            entry = results[key]
            return {
                "kind": "results_json",
                "source": str(results_json),
                "key": key,
                "perplexity": entry.get("perplexity"),
                "cross_entropy": entry.get("cross_entropy"),
                "tokens": entry.get("tokens"),
                "time_seconds": entry.get("time_seconds"),
            }

    return {}


def load_quick_ppl(run_dir: Path, init_metrics: dict[str, Any] | None) -> dict[str, Any]:
    quick = extract_init_quick_ppl(init_metrics or {})
    if quick.get("quick_perplexity") is not None:
        return quick

    init_log = run_dir / "init.log"
    if init_log.exists():
        parsed = parse_init_log_quick_ppl(init_log)
        if parsed.get("quick_perplexity") is not None:
            return parsed

    return quick


def detect_artifact_type(run_dir: Path, init_metrics: dict[str, Any] | None, loss_csv_exists: bool) -> str:
    if init_metrics is not None or (run_dir / "v2_initial.pt").exists():
        return "init"
    if loss_csv_exists:
        return "train"
    return "generic"


def collect_run_summary(
    run_dir: str | Path,
    checkpoint: str | None = None,
    ppl_log: str | None = None,
    results_json: str | None = None,
    baseline_run_dir: str | None = None,
    max_size_growth_pct: float | None = None,
    float_storage_bits: int = 16,
) -> dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"missing run dir: {run_dir}")

    init_metrics_path = run_dir / "init_metrics.json"
    loss_csv = run_dir / "loss.csv"
    config_json = run_dir / "config.json"
    score_json = run_dir / "score.json"

    init_metrics = load_json(init_metrics_path) if init_metrics_path.exists() else None
    config = load_json(config_json) if config_json.exists() else None
    saved_summary = load_json(score_json) if score_json.exists() else None

    rows: list[dict[str, float | None]] = []
    elapsed_sec = None
    if loss_csv.exists():
        rows, elapsed_sec = load_loss_csv(loss_csv)

    eval_losses = [row["eval_loss"] for row in rows if row["eval_loss"] is not None]
    train_losses = [row["train_loss"] for row in rows if row["train_loss"] is not None]
    best_eval_loss = min(eval_losses) if eval_losses else None
    final_eval_loss = eval_losses[-1] if eval_losses else None
    last_train_loss = train_losses[-1] if train_losses else None

    if elapsed_sec is None and init_metrics is not None:
        elapsed_sec = init_metrics.get("total_time_seconds")

    checkpoints = find_candidate_checkpoints(run_dir)
    candidate_checkpoint = find_run_candidate_checkpoint(run_dir, requested_checkpoint=checkpoint)
    if candidate_checkpoint and candidate_checkpoint.exists() and candidate_checkpoint not in checkpoints:
        checkpoints.insert(0, candidate_checkpoint)
    elif candidate_checkpoint is None and saved_summary:
        saved_checkpoint = saved_summary.get("candidate_checkpoint")
        if saved_checkpoint:
            candidate_checkpoint = Path(str(saved_checkpoint))

    quick = load_quick_ppl(run_dir, init_metrics)
    if elapsed_sec is None:
        elapsed_sec = quick.get("elapsed_sec")
    full = load_full_ppl(run_dir, candidate_checkpoint, ppl_log, results_json)

    full_perplexity = full.get("perplexity")
    full_cross_entropy = full.get("cross_entropy")
    full_tokens = full.get("tokens")
    full_source = full.get("source")
    full_key = full.get("key")

    quick_perplexity = quick.get("quick_perplexity")

    score = full_perplexity
    if score is None:
        score = quick_perplexity
    if score is None:
        score = best_eval_loss if best_eval_loss is not None else final_eval_loss
    if score is None:
        score = last_train_loss

    artifact_type = detect_artifact_type(run_dir, init_metrics, loss_csv.exists())
    if candidate_checkpoint and candidate_checkpoint.exists():
        size_summary = estimate_quantized_payload(candidate_checkpoint, float_storage_bits=float_storage_bits)
        size_budget = compare_size_to_baseline(
            candidate_checkpoint,
            candidate_size=size_summary,
            baseline_run_dir=baseline_run_dir,
            max_size_growth_pct=max_size_growth_pct,
            float_storage_bits=float_storage_bits,
        )
    else:
        size_summary = pick_saved_fields(
            saved_summary,
            [
                "size_estimate_mode",
                "size_estimate_error",
                "float_storage_bits",
                "quant_layer_count",
                "quant_weight_count",
                "projected_index_bits",
                "projected_scale_bits",
                "projected_lut_bits",
                "projected_payload_bits",
                "projected_payload_bytes",
                "projected_payload_mib",
                "avg_bits_per_weight",
                "family_avg_bits_per_weight",
            ],
        )
        size_budget = pick_saved_fields(
            saved_summary,
            [
                "baseline_checkpoint",
                "baseline_projected_payload_mib",
                "baseline_avg_bits_per_weight",
                "size_ratio_vs_baseline",
                "size_growth_pct_vs_baseline",
                "size_budget_growth_pct",
                "size_budget_payload_mib",
                "size_budget_avg_bits_per_weight",
                "size_ok",
                "size_budget_error",
            ],
        )

    result = {
        "run_dir": str(run_dir),
        "artifact_type": artifact_type,
        "candidate_checkpoint": str(candidate_checkpoint) if candidate_checkpoint else saved_summary.get("candidate_checkpoint") if saved_summary else None,
        "checkpoints": [str(p) for p in checkpoints],
        "score": score,
        "full_perplexity": full_perplexity,
        "full_cross_entropy": full_cross_entropy,
        "full_tokens": full_tokens,
        "full_ppl_source": full_source,
        "full_ppl_key": full_key,
        "quick_perplexity": quick_perplexity,
        "quick_cross_entropy": quick.get("quick_cross_entropy"),
        "quick_tokens": quick.get("quick_tokens"),
        "best_eval_loss": best_eval_loss,
        "final_eval_loss": final_eval_loss,
        "last_train_loss": last_train_loss,
        "elapsed_sec": elapsed_sec,
        "model_id": (config or {}).get("model_id") or (init_metrics or {}).get("model_id"),
        "config_preset": (config or {}).get("config_preset") or (init_metrics or {}).get("preset"),
        "has_config": config is not None,
        "has_init_metrics": init_metrics is not None,
        "has_loss_csv": loss_csv.exists(),
    }
    result.update(size_summary)
    result.update(size_budget)
    return result


def main() -> int:
    args = parse_args()
    try:
        result = collect_run_summary(
            run_dir=args.run_dir,
            checkpoint=args.checkpoint,
            ppl_log=args.ppl_log,
            results_json=args.results_json,
            baseline_run_dir=args.baseline_run_dir,
            max_size_growth_pct=args.max_size_growth_pct,
            float_storage_bits=args.float_storage_bits,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.format == "json":
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"run_dir:            {result['run_dir']}")
        print(f"artifact_type:      {result['artifact_type']}")
        print(f"candidate_checkpoint:{result['candidate_checkpoint']}")
        print(f"score:              {result['score']}")
        print(f"full_perplexity:    {result['full_perplexity']}")
        print(f"quick_perplexity:   {result['quick_perplexity']}")
        print(f"best_eval_loss:     {result['best_eval_loss']}")
        print(f"final_eval_loss:    {result['final_eval_loss']}")
        print(f"last_train_loss:    {result['last_train_loss']}")
        print(f"elapsed_sec:        {result['elapsed_sec']}")
        print(f"projected_payload_mib:{result.get('projected_payload_mib')}")
        print(f"avg_bits_per_weight:{result.get('avg_bits_per_weight')}")
        print(f"size_ok:            {result.get('size_ok')}")
        print(f"model_id:           {result['model_id']}")
        print(f"config_preset:      {result['config_preset']}")
        print(f"full_ppl_source:    {result['full_ppl_source'] or '-'}")
        print(f"full_ppl_key:       {result['full_ppl_key'] or '-'}")
        print(f"checkpoints:        {', '.join(result['checkpoints']) if result['checkpoints'] else '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
