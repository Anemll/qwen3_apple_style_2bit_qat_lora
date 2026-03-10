#!/usr/bin/env python3
"""Build one AQ1 candidate checkpoint for autoresearch."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aq1_autoresearch.layer_policy import replace_linear_with_layer_overrides
from aq1_autoresearch.score_run import estimate_quantized_payload
from qat_lora.ane_qat_linear_v2 import AnemllQATLinearV2, AnemllQuantConfigV2
from scripts.init_model_v2 import (
    PRESETS,
    create_v2_configs,
    freeze_quantized_weights,
    load_base_model,
    save_checkpoint,
    tighten_and_measure_ppl,
)


PRESET = PRESETS["q4a4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one AQ1 experiment candidate")
    parser.add_argument("--family", choices=["mixedbit", "mixedbit_tiered", "mlp_permute"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--quick-ppl-chunks", type=int, default=14)
    parser.add_argument("--baseline-checkpoint", default=None)
    parser.add_argument("--budget-growth-pct", type=float, default=5.0)
    parser.add_argument("--scope", choices=["all", "mlp", "attn"], default="all")
    parser.add_argument("--selection", choices=["efficiency", "absolute"], default="efficiency")
    parser.add_argument("--upgrade-bits", type=int, default=5)
    parser.add_argument("--max-upgrade-bits", type=int, default=6)
    parser.add_argument(
        "--permute-strategy",
        choices=["combined_desc", "combined_hilo", "down_desc", "down_hilo"],
        default="combined_desc",
    )
    parser.add_argument("--cache-json", default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def write_json(path: Path, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def serialize_layer_overrides(layer_overrides: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, override in layer_overrides.items():
        out[name] = {}
        for key in ("lut_bits", "scale_rank", "group_size"):
            if key in override:
                out[name][key] = int(override[key])
    return out


def iter_quant_linears(model: nn.Module, scope: str = "all") -> list[tuple[str, nn.Linear, str]]:
    layers: list[tuple[str, nn.Linear, str]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if "embed" in name or "lm_head" in name:
            continue
        kind = None
        if ".mlp." in name:
            kind = "mlp"
        elif ".self_attn." in name:
            kind = "attn"
        if kind is None:
            continue
        if scope != "all" and kind != scope:
            continue
        layers.append((name, module, kind))
    return layers


def make_cpu_linear(module: nn.Linear) -> nn.Linear:
    linear = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None)
    with torch.no_grad():
        linear.weight.copy_(module.weight.detach().float().cpu())
        if module.bias is not None:
            linear.bias.copy_(module.bias.detach().float().cpu())
    return linear


@torch.no_grad()
def local_quant_mae(
    module: nn.Linear,
    lut_bits: int,
    scale_rank: int,
    group_size: int,
) -> float:
    linear = make_cpu_linear(module)
    config = AnemllQuantConfigV2(
        lut_size=2 ** lut_bits,
        scale_rank=scale_rank,
        group_size=group_size,
        force_positive_scales=False,
        positive_scale_method="abs",
        magnitude_activation="identity",
        magnitude_eps=0.0,
    )
    v2 = AnemllQATLinearV2.from_linear(linear, config=config, skip_init=False)
    v2.freeze_Q()
    q = v2._Q.view(v2.out_features, v2.in_features)
    s = v2._compute_full_scales()
    w_ref = linear.weight.data.float()
    return float((w_ref - (q * s)).abs().mean().item())


def layer_payload_bits(module: nn.Linear, lut_bits: int, scale_rank: int, float_storage_bits: int = 16) -> int:
    weight_count = module.in_features * module.out_features
    index_bits = weight_count * lut_bits
    scale_bits = (module.out_features * scale_rank + scale_rank * module.in_features + scale_rank) * float_storage_bits
    lut_bits_total = (2 ** lut_bits) * float_storage_bits
    return index_bits + scale_bits + lut_bits_total


def load_or_compute_layer_stats(
    model: nn.Module,
    group_size: int,
    scope: str,
    bit_options: list[int],
    cache_json: Path | None,
) -> list[dict[str, Any]]:
    normalized_bits = sorted({int(bits) for bits in bit_options})
    cache_key = f"{scope}|{group_size}|{','.join(str(bits) for bits in normalized_bits)}"

    if cache_json and cache_json.exists():
        cached = json.loads(cache_json.read_text(encoding="utf-8"))
        if cached.get("cache_key") == cache_key and isinstance(cached.get("candidates"), list):
            return cached["candidates"]

    candidates: list[dict[str, Any]] = []
    for name, module, kind in iter_quant_linears(model, scope=scope):
        scale_rank = PRESET.attn_rank if kind == "attn" else PRESET.mlp_rank
        mae_by_bits: dict[str, float] = {}
        payload_bits_by_bits: dict[str, int] = {}
        for lut_bits in normalized_bits:
            mae_by_bits[str(lut_bits)] = local_quant_mae(
                module,
                lut_bits=lut_bits,
                scale_rank=scale_rank,
                group_size=group_size,
            )
            payload_bits_by_bits[str(lut_bits)] = layer_payload_bits(
                module,
                lut_bits=lut_bits,
                scale_rank=scale_rank,
            )
        candidates.append(
            {
                "name": name,
                "kind": kind,
                "scale_rank": scale_rank,
                "mae_by_bits": mae_by_bits,
                "payload_bits_by_bits": payload_bits_by_bits,
            }
        )

    if cache_json:
        cache_json.parent.mkdir(parents=True, exist_ok=True)
        write_json(cache_json, {"cache_key": cache_key, "candidates": candidates})
    return candidates


def build_mixedbit_overrides(
    model: nn.Module,
    baseline_checkpoint: Path,
    budget_growth_pct: float,
    group_size: int,
    scope: str,
    selection: str,
    upgrade_bits: int,
    cache_json: Path | None,
    verbose: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    baseline_size = estimate_quantized_payload(baseline_checkpoint)
    baseline_bits = baseline_size.get("projected_payload_bits")
    if baseline_bits is None:
        raise RuntimeError(f"failed to estimate baseline size from {baseline_checkpoint}")

    extra_budget_bits = int(baseline_bits * (budget_growth_pct / 100.0))
    raw_candidates = load_or_compute_layer_stats(
        model=model,
        group_size=group_size,
        scope=scope,
        bit_options=[4, upgrade_bits],
        cache_json=cache_json,
    )
    candidates: list[dict[str, Any]] = []
    for row in raw_candidates:
        base_mae = row["mae_by_bits"]["4"]
        upgrade_mae = row["mae_by_bits"][str(upgrade_bits)]
        extra_bits = row["payload_bits_by_bits"][str(upgrade_bits)] - row["payload_bits_by_bits"]["4"]
        improvement = base_mae - upgrade_mae
        candidates.append(
            {
                "name": row["name"],
                "kind": row["kind"],
                "scale_rank": row["scale_rank"],
                "base_mae": base_mae,
                "upgrade_mae": upgrade_mae,
                "improvement": improvement,
                "extra_bits": extra_bits,
                "efficiency": (improvement / extra_bits) if extra_bits > 0 else 0.0,
            }
        )

    if selection == "absolute":
        ranked = sorted(candidates, key=lambda row: row["improvement"], reverse=True)
    else:
        ranked = sorted(candidates, key=lambda row: row["efficiency"], reverse=True)

    selected: list[dict[str, Any]] = []
    used_bits = 0
    for row in ranked:
        if row["improvement"] <= 0 or row["extra_bits"] <= 0:
            continue
        if used_bits + row["extra_bits"] > extra_budget_bits:
            continue
        selected.append(row)
        used_bits += row["extra_bits"]

    layer_overrides = {
        row["name"]: {
            "lut_bits": upgrade_bits,
            "scale_rank": row["scale_rank"],
            "group_size": group_size,
        }
        for row in selected
    }

    metadata = {
        "family": "mixedbit",
        "scope": scope,
        "selection": selection,
        "upgrade_bits": upgrade_bits,
        "budget_growth_pct": budget_growth_pct,
        "baseline_payload_mib": baseline_size.get("projected_payload_mib"),
        "baseline_avg_bits_per_weight": baseline_size.get("avg_bits_per_weight"),
        "extra_budget_bits": extra_budget_bits,
        "used_extra_bits": used_bits,
        "selected_layers": len(selected),
        "selected_names": [row["name"] for row in selected],
        "top_candidates": ranked[:12],
    }

    if verbose:
        print(f"[mixedbit] Selected {len(selected)} layers, used {used_bits:,}/{extra_budget_bits:,} extra bits")

    return layer_overrides, metadata


def build_tiered_mixedbit_overrides(
    model: nn.Module,
    baseline_checkpoint: Path,
    budget_growth_pct: float,
    group_size: int,
    scope: str,
    selection: str,
    max_upgrade_bits: int,
    cache_json: Path | None,
    verbose: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    baseline_size = estimate_quantized_payload(baseline_checkpoint)
    baseline_bits = baseline_size.get("projected_payload_bits")
    if baseline_bits is None:
        raise RuntimeError(f"failed to estimate baseline size from {baseline_checkpoint}")
    if max_upgrade_bits < 5:
        raise RuntimeError(f"expected max_upgrade_bits >= 5, got {max_upgrade_bits}")

    extra_budget_bits = int(baseline_bits * (budget_growth_pct / 100.0))
    candidates = load_or_compute_layer_stats(
        model=model,
        group_size=group_size,
        scope=scope,
        bit_options=list(range(4, max_upgrade_bits + 1)),
        cache_json=cache_json,
    )

    current_bits = {row["name"]: 4 for row in candidates}
    current_mae = {row["name"]: float(row["mae_by_bits"]["4"]) for row in candidates}
    payload_bits = {row["name"]: {int(bits): int(value) for bits, value in row["payload_bits_by_bits"].items()} for row in candidates}
    mae_by_bits = {row["name"]: {int(bits): float(value) for bits, value in row["mae_by_bits"].items()} for row in candidates}
    row_by_name = {row["name"]: row for row in candidates}

    selected_actions: list[dict[str, Any]] = []
    used_bits = 0
    while True:
        remaining_bits = extra_budget_bits - used_bits
        best_action: dict[str, Any] | None = None

        for row in candidates:
            name = row["name"]
            cur_bits = current_bits[name]
            next_bits = cur_bits + 1
            if next_bits > max_upgrade_bits:
                continue

            extra_bits = payload_bits[name][next_bits] - payload_bits[name][cur_bits]
            if extra_bits <= 0 or extra_bits > remaining_bits:
                continue

            improvement = current_mae[name] - mae_by_bits[name][next_bits]
            if improvement <= 0:
                continue

            score = improvement if selection == "absolute" else improvement / extra_bits
            action = {
                "name": name,
                "kind": row["kind"],
                "from_bits": cur_bits,
                "to_bits": next_bits,
                "improvement": improvement,
                "extra_bits": extra_bits,
                "score": score,
            }
            if best_action is None or (action["score"], action["improvement"]) > (
                best_action["score"],
                best_action["improvement"],
            ):
                best_action = action

        if best_action is None:
            break

        name = best_action["name"]
        current_bits[name] = best_action["to_bits"]
        current_mae[name] = mae_by_bits[name][best_action["to_bits"]]
        used_bits += best_action["extra_bits"]
        selected_actions.append(best_action)

    layer_overrides = {
        name: {
            "lut_bits": bits,
            "scale_rank": row_by_name[name]["scale_rank"],
            "group_size": group_size,
        }
        for name, bits in current_bits.items()
        if bits > 4
    }

    selected_names = sorted(layer_overrides)
    bit_histogram: dict[int, int] = {}
    for bits in layer_overrides.values():
        lut_bits = int(bits["lut_bits"])
        bit_histogram[lut_bits] = bit_histogram.get(lut_bits, 0) + 1

    metadata = {
        "family": "mixedbit_tiered",
        "scope": scope,
        "selection": selection,
        "max_upgrade_bits": max_upgrade_bits,
        "budget_growth_pct": budget_growth_pct,
        "baseline_payload_mib": baseline_size.get("projected_payload_mib"),
        "baseline_avg_bits_per_weight": baseline_size.get("avg_bits_per_weight"),
        "extra_budget_bits": extra_budget_bits,
        "used_extra_bits": used_bits,
        "selected_layers": len(selected_names),
        "selected_names": selected_names,
        "selected_lut_histogram": bit_histogram,
        "upgrade_actions": selected_actions[:32],
    }

    if verbose:
        hist = ", ".join(f"{bits}-bit:{count}" for bits, count in sorted(bit_histogram.items()))
        print(
            f"[mixedbit_tiered] Selected {len(selected_names)} layers, used "
            f"{used_bits:,}/{extra_budget_bits:,} extra bits"
            + (f" ({hist})" if hist else "")
        )

    return layer_overrides, metadata


def interleave_high_low(indices: torch.Tensor) -> torch.Tensor:
    values = indices.tolist()
    out: list[int] = []
    left = 0
    right = len(values) - 1
    take_high = True
    while left <= right:
        if take_high:
            out.append(values[left])
            left += 1
        else:
            out.append(values[right])
            right -= 1
        take_high = not take_high
    return torch.tensor(out, dtype=torch.long, device=indices.device)


@torch.no_grad()
def apply_mlp_hidden_permutations(model: nn.Module, strategy: str, verbose: bool) -> dict[str, Any]:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError("expected HuggingFace decoder model with model.layers")

    layer_summaries: list[dict[str, Any]] = []
    for layer_idx, block in enumerate(model.model.layers):
        gate = block.mlp.gate_proj
        up = block.mlp.up_proj
        down = block.mlp.down_proj

        gate_score = gate.weight.detach().abs().mean(dim=1)
        up_score = up.weight.detach().abs().mean(dim=1)
        down_score = down.weight.detach().abs().mean(dim=0)
        if strategy.startswith("down_"):
            score = down_score
        else:
            score = gate_score + up_score + down_score

        perm = torch.argsort(score, descending=True)
        if strategy.endswith("_hilo"):
            perm = interleave_high_low(perm)

        inv_perm = torch.argsort(perm)

        gate.weight.data = gate.weight.data[perm]
        up.weight.data = up.weight.data[perm]
        if gate.bias is not None:
            gate.bias.data = gate.bias.data[perm]
        if up.bias is not None:
            up.bias.data = up.bias.data[perm]
        down.weight.data = down.weight.data[:, inv_perm]

        layer_summaries.append(
            {
                "layer": layer_idx,
                "score_min": float(score.min().item()),
                "score_max": float(score.max().item()),
                "score_mean": float(score.mean().item()),
            }
        )

    if verbose:
        print(f"[mlp_permute] Applied strategy {strategy} to {len(layer_summaries)} MLP blocks")

    return {
        "family": "mlp_permute",
        "strategy": strategy,
        "num_layers": len(layer_summaries),
        "layer_summaries": layer_summaries[:8],
    }


def build_tightened_model(
    model_id: str,
    checkpoint_path: Path,
    layer_overrides: dict[str, dict[str, Any]],
    group_size: int,
) -> AutoModelForCausalLM:
    mlp_config, attn_config = create_v2_configs(PRESET, group_size=group_size, verbose=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    replace_linear_with_layer_overrides(
        model,
        mlp_config=mlp_config,
        attn_config=attn_config,
        layer_overrides=layer_overrides,
        quantize_attn=True,
        verbose=False,
        skip_init=True,
    )

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    q_loaded = 0
    for name, module in model.named_modules():
        if isinstance(module, AnemllQATLinearV2):
            q_key = f"{name}._Q"
            if q_key in state_dict and module._Q is None:
                module.register_buffer("_Q", state_dict[q_key])
                q_loaded += 1
    if q_loaded:
        print(f"[tighten] Loaded {q_loaded} _Q buffers")
    return model


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    verbose = not args.quiet
    layer_overrides: dict[str, dict[str, Any]] = {}
    experiment_metadata: dict[str, Any] = {
        "family": args.family,
        "group_size": args.group_size,
    }

    model, tokenizer = load_base_model(
        model_id=args.model_id,
        device=device,
        dtype=torch.float32,
        verbose=verbose,
    )

    if args.family in {"mixedbit", "mixedbit_tiered"}:
        if not args.baseline_checkpoint:
            raise SystemExit("--baseline-checkpoint is required for mixed-bit experiments")
        cache_json = Path(args.cache_json).expanduser().resolve() if args.cache_json else None
        if args.family == "mixedbit":
            layer_overrides, family_meta = build_mixedbit_overrides(
                model=model,
                baseline_checkpoint=Path(args.baseline_checkpoint).expanduser().resolve(),
                budget_growth_pct=args.budget_growth_pct,
                group_size=args.group_size,
                scope=args.scope,
                selection=args.selection,
                upgrade_bits=args.upgrade_bits,
                cache_json=cache_json,
                verbose=verbose,
            )
        else:
            layer_overrides, family_meta = build_tiered_mixedbit_overrides(
                model=model,
                baseline_checkpoint=Path(args.baseline_checkpoint).expanduser().resolve(),
                budget_growth_pct=args.budget_growth_pct,
                group_size=args.group_size,
                scope=args.scope,
                selection=args.selection,
                max_upgrade_bits=args.max_upgrade_bits,
                cache_json=cache_json,
                verbose=verbose,
            )
        experiment_metadata.update(family_meta)
    else:
        experiment_metadata.update(apply_mlp_hidden_permutations(model, strategy=args.permute_strategy, verbose=verbose))

    mlp_config, attn_config = create_v2_configs(
        preset=PRESET,
        group_size=args.group_size,
        verbose=verbose,
    )

    replace_linear_with_layer_overrides(
        model,
        mlp_config=mlp_config,
        attn_config=attn_config,
        layer_overrides=layer_overrides,
        quantize_attn=True,
        verbose=verbose,
        skip_init=False,
    )
    model.to(device)

    metrics: dict[str, Any] = {
        "model_id": args.model_id,
        "preset": PRESET.name,
        "device": str(device),
        "steps": {},
        "family": args.family,
        "experiment_metadata": experiment_metadata,
    }

    freeze_stats = freeze_quantized_weights(model, verbose=verbose)
    metrics["steps"]["freeze"] = freeze_stats

    saved = save_checkpoint(
        model=model,
        output_dir=str(output_dir),
        preset=PRESET,
        model_id=args.model_id,
        group_size=args.group_size,
        init_metrics=metrics,
        layer_overrides=serialize_layer_overrides(layer_overrides) or None,
        experiment_metadata=experiment_metadata,
        verbose=verbose,
    )

    checkpoint_path = Path(saved["checkpoint"])
    tightened_model = build_tightened_model(
        model_id=args.model_id,
        checkpoint_path=checkpoint_path,
        layer_overrides=layer_overrides,
        group_size=args.group_size,
    )

    tighten_results = tighten_and_measure_ppl(
        model=tightened_model,
        tokenizer=tokenizer,
        model_id=args.model_id,
        device=device,
        num_chunks=args.quick_ppl_chunks,
        verbose=verbose,
        skip_ppl=False,
    )
    metrics["steps"]["perplexity"] = tighten_results
    metrics["total_time_seconds"] = tighten_results.get("time_seconds")

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    tightened_path = output_dir / "v2_tightened.pt"
    torch.save(tightened_model.state_dict(), tightened_path)

    write_json(Path(saved["metrics"]), metrics)
    write_json(output_dir / "candidate_plan.json", {"args": vars(args), "experiment_metadata": experiment_metadata})

    size_summary = estimate_quantized_payload(tightened_path)
    if size_summary.get("avg_bits_per_weight") is not None:
        print(
            f"[size] projected={size_summary['projected_payload_mib']:.2f} MiB "
            f"avg_bits={size_summary['avg_bits_per_weight']:.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
