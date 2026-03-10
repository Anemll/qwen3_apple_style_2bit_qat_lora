#!/usr/bin/env python3
"""Render an AQ1 progress chart and markdown summary from results.tsv."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import textwrap
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO_ROOT / "aq1_autoresearch" / "results.tsv"
DEFAULT_CAMPAIGN = REPO_ROOT / "aq1_autoresearch" / "campaign.md"
DEFAULT_RUNNER_STATE = REPO_ROOT / "runs" / "aq1_auto" / "qwen06b-init-ppl" / "auto6h_20260310_run8" / "runner_state.json"
DEFAULT_PERPLEXITY_JSON = REPO_ROOT / "results" / "perplexity.json"
DEFAULT_SVG = REPO_ROOT / "aq1_autoresearch" / "progress_report.svg"
DEFAULT_MD = REPO_ROOT / "aq1_autoresearch" / "progress_report.md"
REPORT_TITLE = "AQ1 ANE-Native Quantization Experiments"
REPORT_SUBTITLE = "Apple Neural Engine native quantization search; full perplexity is the keep metric and quick-only points are screens"
REPORT_MD_TITLE = "# AQ1 ANE-Native Quantization Summary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render AQ1 progress SVG + markdown summary")
    parser.add_argument("--results-tsv", default=str(DEFAULT_RESULTS))
    parser.add_argument("--campaign", default=str(DEFAULT_CAMPAIGN))
    parser.add_argument("--runner-state", default=str(DEFAULT_RUNNER_STATE))
    parser.add_argument("--perplexity-json", default=str(DEFAULT_PERPLEXITY_JSON))
    parser.add_argument("--output-svg", default=str(DEFAULT_SVG))
    parser.add_argument("--output-md", default=str(DEFAULT_MD))
    return parser.parse_args()


def maybe_float(value: str | None) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    return float(value)


def load_rows(results_tsv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(results_tsv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        best_full = float("inf")
        for idx, row in enumerate(reader):
            full = maybe_float(row.get("full_perplexity"))
            quick = maybe_float(row.get("quick_perplexity"))
            metric = full if full is not None else quick
            metric_kind = "full" if full is not None else "quick"
            run_name = Path((row.get("run_dir") or "").strip()).name or f"row_{idx}"
            new_best = full is not None and full < best_full
            if new_best:
                best_full = full
            rows.append(
                {
                    "index": idx,
                    "row": row,
                    "run_name": run_name,
                    "metric": metric,
                    "metric_kind": metric_kind,
                    "full": full,
                    "quick": quick,
                    "status": (row.get("status") or "").strip(),
                    "change_family": (row.get("change_family") or "").strip(),
                    "description": (row.get("description") or "").strip(),
                    "avg_bits": maybe_float(row.get("avg_bits_per_weight")),
                    "size_mib": maybe_float(row.get("projected_payload_mib")),
                    "new_best": new_best,
                }
            )
    return rows


def load_runner_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_campaign_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    if not path.exists():
        return metadata
    bullet = re.compile(r"^- (?P<key>[^:]+): (?P<value>.+)$")
    with open(path, encoding="utf-8") as f:
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


def load_original_reference(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    entry = data.get("baseline:Qwen/Qwen3-0.6B")
    if not isinstance(entry, dict) or entry.get("perplexity") is None:
        return None
    return {
        "perplexity": float(entry["perplexity"]),
        "cross_entropy": entry.get("cross_entropy"),
        "tokens": entry.get("tokens"),
        "time_seconds": entry.get("time_seconds"),
        "dtype": entry.get("dtype"),
        "dataset": entry.get("dataset"),
    }


def running_best_series(rows: list[dict[str, Any]]) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    best = float("inf")
    last = None
    for row in rows:
        if row["full"] is not None and row["full"] < best:
            best = row["full"]
        if best < float("inf"):
            last = best
        if last is not None:
            points.append((row["index"], last))
    return points


def svg_escape(text: str) -> str:
    return html.escape(text, quote=True)


def wrap_svg_text(text: str, width: int) -> list[str]:
    if not text:
        return []
    return textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)


def shorten_svg_text(text: str, width: int) -> str:
    if not text:
        return ""
    return textwrap.shorten(text, width=width, placeholder="...")


def choose_y_bounds(rows: list[dict[str, Any]], baseline: float, best: float) -> tuple[float, float]:
    hard_cap = max(baseline * 1.25, best + 6.0)
    in_range = [row["metric"] for row in rows if row["metric"] is not None and row["metric"] <= hard_cap]
    if not in_range:
        in_range = [baseline, best]
    y_min = min(in_range) - 0.6
    y_max = max(in_range) + 0.8
    if y_max - y_min < 4.0:
        mid = 0.5 * (y_min + y_max)
        y_min = mid - 2.0
        y_max = mid + 2.0
    return y_min, y_max


def render_svg(
    rows: list[dict[str, Any]],
    campaign: dict[str, str],
    runner_state: dict[str, Any] | None,
    original_ref: dict[str, Any] | None,
    output_svg: Path,
) -> None:
    baseline = rows[0]["full"] if rows and rows[0]["full"] is not None else min(row["full"] for row in rows if row["full"] is not None)
    best_row = min((row for row in rows if row["full"] is not None), key=lambda row: row["full"])
    new_best_rows = [row for row in rows if row["new_best"]]
    best = best_row["full"]
    y_min, y_max = choose_y_bounds(rows, baseline=baseline, best=best)

    width = 1280
    height = 860
    left = 90
    right = 40
    top = 250
    bottom = 95
    plot_w = width - left - right
    plot_h = height - top - bottom
    n = max(len(rows), 2)

    def x_pos(index: int) -> float:
        if n <= 1:
            return left + plot_w / 2
        return left + (plot_w * index / (n - 1))

    def y_pos(value: float) -> float:
        clipped = max(min(value, y_max), y_min)
        return top + plot_h * (1.0 - (clipped - y_min) / (y_max - y_min))

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<rect width="100%" height="100%" fill="#fbf7ef"/>')
    parts.append('<rect x="0" y="0" width="100%" height="100%" fill="url(#bg)"/>')
    parts.append(
        "<defs>"
        '<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0%" stop-color="#fff9ef"/>'
        '<stop offset="100%" stop-color="#f3efe6"/>'
        "</linearGradient>"
        "</defs>"
    )

    parts.append(f'<text x="{left}" y="46" font-family="Menlo, monospace" font-size="28" font-weight="700" fill="#1d2733">{svg_escape(REPORT_TITLE)}</text>')
    parts.append(f'<text x="{left}" y="72" font-family="Menlo, monospace" font-size="14" fill="#5b6470">{svg_escape(REPORT_SUBTITLE)}</text>')
    model_id = campaign.get("model_id")
    config_preset = campaign.get("config_preset")
    baseline_bits = campaign.get("baseline_average_bits_per_weight")
    ceiling_bits = campaign.get("average_bits_per_weight_ceiling")
    metadata_parts = []
    if model_id:
        metadata_parts.append(f"model {model_id}")
    if config_preset:
        metadata_parts.append(f"target {config_preset}")
    if baseline_bits:
        metadata_parts.append(f"baseline {baseline_bits} bpw")
    if ceiling_bits:
        metadata_parts.append(f"ceiling {ceiling_bits} bpw")
    if metadata_parts:
        parts.append(
            f'<text x="{left}" y="94" font-family="Menlo, monospace" font-size="13" fill="#5b6470">'
            f'{svg_escape(" | ".join(metadata_parts))}</text>'
        )

    if runner_state:
        state_text = (
            f"live run: {runner_state.get('completed_experiments', 0)} done, "
            f"best {runner_state.get('best_full_perplexity')}, "
            f"elapsed {runner_state.get('elapsed_hours', 0):.2f}h"
        )
        parts.append(f'<text x="{width - right}" y="46" text-anchor="end" font-family="Menlo, monospace" font-size="13" fill="#5b6470">{svg_escape(state_text)}</text>')

    for step in range(6):
        y_value = y_min + step * (y_max - y_min) / 5.0
        y = y_pos(y_value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#d9d2c8" stroke-width="1"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Menlo, monospace" font-size="12" fill="#5b6470">{y_value:.1f}</text>')

    x_ticks = min(8, len(rows))
    for tick in range(x_ticks):
        index = round(tick * (len(rows) - 1) / max(x_ticks - 1, 1))
        x = x_pos(index)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#ece6dc" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - bottom + 24}" text-anchor="middle" font-family="Menlo, monospace" font-size="12" fill="#5b6470">{index}</text>')

    parts.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#1d2733" stroke-width="1.5"/>')

    baseline_y = y_pos(baseline)
    parts.append(f'<line x1="{left}" y1="{baseline_y:.1f}" x2="{width - right}" y2="{baseline_y:.1f}" stroke="#5f6b76" stroke-width="1.5" stroke-dasharray="6 6"/>')
    parts.append(f'<text x="{width - right}" y="{baseline_y - 8:.1f}" text-anchor="end" font-family="Menlo, monospace" font-size="12" fill="#5f6b76">baseline {baseline:.2f}</text>')
    if original_ref is not None:
        original_y = y_pos(float(original_ref["perplexity"]))
        parts.append(f'<line x1="{left}" y1="{original_y:.1f}" x2="{width - right}" y2="{original_y:.1f}" stroke="#2f855a" stroke-width="1.5" stroke-dasharray="3 5"/>')
        parts.append(
            f'<text x="{width - right}" y="{original_y - 8:.1f}" text-anchor="end" '
            f'font-family="Menlo, monospace" font-size="12" fill="#2f855a">original {float(original_ref["perplexity"]):.2f}</text>'
        )

    best_points = running_best_series(rows)
    if best_points:
        d = []
        for idx, value in best_points:
            x = x_pos(idx)
            y = y_pos(value)
            d.append(f"L {x:.1f} {y:.1f}" if d else f"M {x:.1f} {y:.1f}")
        parts.append(f'<path d="{" ".join(d)}" fill="none" stroke="#0f1720" stroke-width="3"/>')

    off_scale_count = 0
    for row in rows:
        metric = row["metric"]
        if metric is None:
            continue
        x = x_pos(row["index"])
        if metric > y_max:
            off_scale_count += 1
            y = y_pos(y_max)
            parts.append(f'<path d="M {x - 5:.1f} {y - 7:.1f} L {x:.1f} {y - 2:.1f} L {x + 5:.1f} {y - 7:.1f} Z" fill="#8b1e1e"/>')
            continue

        y = y_pos(metric)
        if row["metric_kind"] == "quick":
            color = "#c46a00"
            parts.append(f'<line x1="{x - 4:.1f}" y1="{y - 4:.1f}" x2="{x + 4:.1f}" y2="{y + 4:.1f}" stroke="{color}" stroke-width="2"/>')
            parts.append(f'<line x1="{x - 4:.1f}" y1="{y + 4:.1f}" x2="{x + 4:.1f}" y2="{y - 4:.1f}" stroke="{color}" stroke-width="2"/>')
        else:
            if row["status"] == "keep":
                fill = "#0c7c59"
            else:
                fill = "#d55d3f"
            radius = 6 if row["new_best"] else 4.5
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{fill}" stroke="#ffffff" stroke-width="1.5"/>')
            if row["new_best"]:
                parts.append(
                    f'<text x="{x + 8:.1f}" y="{y - 8:.1f}" font-family="Menlo, monospace" font-size="11" '
                    f'font-weight="700" fill="#114b5f">#{row["index"]:02d}</text>'
                )

    best_x = x_pos(best_row["index"])
    best_y = y_pos(best_row["full"])
    parts.append(f'<circle cx="{best_x:.1f}" cy="{best_y:.1f}" r="7.5" fill="#114b5f" stroke="#ffffff" stroke-width="2"/>')
    parts.append(
        f'<text x="{best_x + 10:.1f}" y="{best_y - 10:.1f}" font-family="Menlo, monospace" '
        f'font-size="12" font-weight="700" fill="#114b5f">{best_row["full"]:.2f}</text>'
    )

    best_improvement = baseline - best_row["full"]
    best_gap = None
    if original_ref is not None:
        best_gap = best_row["full"] - float(original_ref["perplexity"])
    detail_lines = [
        best_row["run_name"],
        f"PPL {best_row['full']:.2f} | gain vs q4 {best_improvement:.2f}",
    ]
    if best_gap is not None:
        detail_lines.append(f"gap to original {best_gap:.2f}")
    if best_row["size_mib"] is not None and best_row["avg_bits"] is not None:
        detail_lines.append(f"{best_row['size_mib']:.2f} MiB | {best_row['avg_bits']:.4f} bpw")
    detail_lines.extend(wrap_svg_text(best_row["description"], width=46))

    callout_w = 400
    callout_x = width - right - callout_w
    callout_y = 108
    callout_h = 22 + 20 + len(detail_lines) * 16 + 18
    connector_x = callout_x + 24
    connector_y = callout_y + callout_h
    parts.append(
        f'<line x1="{connector_x:.1f}" y1="{connector_y:.1f}" x2="{best_x:.1f}" y2="{best_y:.1f}" '
        f'stroke="#114b5f" stroke-width="1.5" stroke-dasharray="5 4" opacity="0.7"/>'
    )
    parts.append(
        f'<rect x="{callout_x}" y="{callout_y}" width="{callout_w}" height="{callout_h}" rx="14" ry="14" '
        f'fill="#fffaf0" stroke="#114b5f" stroke-width="2"/>'
    )
    parts.append(
        f'<text x="{callout_x + 18}" y="{callout_y + 26}" font-family="Menlo, monospace" font-size="15" '
        f'font-weight="700" fill="#114b5f">Best New Result</text>'
    )
    line_y = callout_y + 50
    for idx, line in enumerate(detail_lines):
        font_weight = "700" if idx == 0 else "400"
        fill = "#1d2733" if idx == 0 else "#374151"
        parts.append(
            f'<text x="{callout_x + 18}" y="{line_y}" font-family="Menlo, monospace" font-size="12.5" '
            f'font-weight="{font_weight}" fill="{fill}">{svg_escape(line)}</text>'
        )
        line_y += 16

    ladder_x = left
    ladder_y = 108
    ladder_w = callout_x - ladder_x - 20
    ladder_lines: list[str] = []
    prev_best = None
    for row in new_best_rows:
        if prev_best is None:
            gain_text = "seed"
        else:
            gain_text = f"+{prev_best - row['full']:.2f}"
        short_desc = shorten_svg_text(row["description"], width=56)
        ladder_lines.append(f"#{row['index']:02d} {row['full']:.2f} | {gain_text} | {short_desc}")
        prev_best = row["full"]
    ladder_h = 22 + 20 + len(ladder_lines) * 16 + 18
    parts.append(
        f'<rect x="{ladder_x}" y="{ladder_y}" width="{ladder_w}" height="{ladder_h}" rx="14" ry="14" '
        f'fill="#fffaf0" stroke="#4a5568" stroke-width="1.6"/>'
    )
    parts.append(
        f'<text x="{ladder_x + 18}" y="{ladder_y + 26}" font-family="Menlo, monospace" font-size="15" '
        f'font-weight="700" fill="#1d2733">Improvement Steps</text>'
    )
    parts.append(
        f'<text x="{ladder_x + 18}" y="{ladder_y + 44}" font-family="Menlo, monospace" font-size="11.5" '
        f'fill="#5b6470">Each line is a new best. Labels on the plot use the same #index.</text>'
    )
    ladder_line_y = ladder_y + 66
    for line in ladder_lines:
        parts.append(
            f'<text x="{ladder_x + 18}" y="{ladder_line_y}" font-family="Menlo, monospace" font-size="11.5" '
            f'fill="#374151">{svg_escape(line)}</text>'
        )
        ladder_line_y += 16

    legend_x = left
    legend_y = height - 42
    legend = [
        ('<circle cx="0" cy="0" r="5" fill="#0c7c59"/>', "new kept best"),
        ('<circle cx="0" cy="0" r="5" fill="#d55d3f"/>', "full eval, discarded"),
        ('<path d="M -5 -5 L 5 5 M -5 5 L 5 -5" stroke="#c46a00" stroke-width="2"/>', "quick-only screen"),
        ('<path d="M 0 -5 L 5 3 L -5 3 Z" fill="#8b1e1e"/>', "off-scale failure"),
        ('<line x1="-7" y1="0" x2="7" y2="0" stroke="#0f1720" stroke-width="3"/>', "running best"),
        ('<line x1="-7" y1="0" x2="7" y2="0" stroke="#2f855a" stroke-width="2" stroke-dasharray="3 5"/>', "original model"),
    ]
    cursor_x = legend_x
    for mark, label in legend:
        parts.append(f'<g transform="translate({cursor_x},{legend_y})">{mark}</g>')
        parts.append(f'<text x="{cursor_x + 14}" y="{legend_y + 4}" font-family="Menlo, monospace" font-size="12" fill="#5b6470">{svg_escape(label)}</text>')
        cursor_x += 175

    if off_scale_count:
        parts.append(f'<text x="{width - right}" y="{legend_y + 4}" text-anchor="end" font-family="Menlo, monospace" font-size="12" fill="#8b1e1e">{off_scale_count} off-scale failures clipped</text>')

    parts.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 18}" text-anchor="middle" font-family="Menlo, monospace" font-size="13" fill="#1d2733">experiment order</text>')
    parts.append(
        f'<text x="24" y="{top + plot_h / 2:.1f}" transform="rotate(-90 24 {top + plot_h / 2:.1f})" '
        'text-anchor="middle" font-family="Menlo, monospace" font-size="13" fill="#1d2733">perplexity (lower is better)</text>'
    )
    parts.append("</svg>")

    output_svg.write_text("\n".join(parts), encoding="utf-8")


def summarize(
    rows: list[dict[str, Any]],
    campaign: dict[str, str],
    runner_state: dict[str, Any] | None,
    original_ref: dict[str, Any] | None,
) -> str:
    baseline = rows[0]["full"]
    best_row = min((row for row in rows if row["full"] is not None), key=lambda row: row["full"])
    best = best_row["full"]
    improvement = baseline - best
    improvement_pct = improvement / baseline * 100.0
    original_ppl = float(original_ref["perplexity"]) if original_ref is not None else None
    quant_penalty = baseline - original_ppl if original_ppl is not None else None
    best_gap = best - original_ppl if original_ppl is not None else None
    recovered_pct = (improvement / quant_penalty * 100.0) if quant_penalty and quant_penalty > 0 else None

    full_rows = [row for row in rows if row["full"] is not None]
    quick_only_rows = [row for row in rows if row["full"] is None and row["quick"] is not None]
    new_best_rows = [row for row in rows if row["new_best"]]

    family_best: dict[str, tuple[float, dict[str, Any]]] = {}
    for row in full_rows:
        family = row["change_family"]
        current = family_best.get(family)
        if current is None or row["full"] < current[0]:
            family_best[family] = (row["full"], row)

    ranked = sorted((row for row in full_rows), key=lambda row: row["full"])[:6]

    lines = [
        REPORT_MD_TITLE,
        "",
        f"- Model id: {campaign.get('model_id', 'unknown')}",
        f"- Quant target: {campaign.get('config_preset', 'unknown')}",
        f"- Baseline / ceiling bits: {campaign.get('baseline_average_bits_per_weight', 'unknown')} / {campaign.get('average_bits_per_weight_ceiling', 'unknown')} bpw",
        f"- Completed rows logged: {len(rows)}",
        f"- Full perplexity evaluations: {len(full_rows)}",
        f"- Quick-screen-only rows: {len(quick_only_rows)}",
        f"- Baseline full perplexity: {baseline:.2f}",
        f"- Best full perplexity: {best:.2f} ({best_row['run_name']})",
        f"- Improvement vs baseline: {improvement:.2f} ({improvement_pct:.2f}%)",
        f"- Best payload / avg bits: {best_row['size_mib']:.2f} MiB / {best_row['avg_bits']:.4f} bits per weight",
        f"- Original model perplexity: {original_ppl:.2f} (Qwen/Qwen3-0.6B, CPU fp32, same max_chunks=20)" if original_ppl is not None else "- Original model perplexity: unavailable",
        f"- Quantized baseline delta vs original: +{quant_penalty:.2f}" if quant_penalty is not None else "- Quantized baseline delta vs original: unavailable",
        f"- Best run delta vs original: +{best_gap:.2f}" if best_gap is not None else "- Best run delta vs original: unavailable",
        f"- Recovered quantization gap: {improvement:.2f} / {quant_penalty:.2f} ({recovered_pct:.2f}%)" if recovered_pct is not None else "- Recovered quantization gap: unavailable",
        "",
        "## New Best Sequence",
        "",
    ]
    for row in new_best_rows:
        lines.append(f"- #{row['index']:02d} {row['run_name']}: {row['full']:.2f} [{row['description']}]")

    lines.extend(
        [
            "",
            "## Best By Family",
            "",
        ]
    )
    for family, (_, row) in sorted(family_best.items(), key=lambda item: item[1][0]):
        lines.append(f"- {family}: {row['full']:.2f} ({row['run_name']})")

    lines.extend(
        [
            "",
            "## Best Overall Runs",
            "",
        ]
    )
    for row in ranked:
        delta_q4 = row["full"] - baseline
        delta_orig = row["full"] - original_ppl if original_ppl is not None else None
        lines.append(
            f"- {row['full']:.2f} | {row['run_name']} | {row['description']} | "
            f"{row['size_mib']:.2f} MiB | {row['avg_bits']:.4f} bits | delta q4 {delta_q4:+.2f}"
            + (f" | delta orig {delta_orig:+.2f}" if delta_orig is not None else "")
        )

    lines.extend(
        [
            "",
            "## Current Read",
            "",
            "- g16 is clearly better than g32 in this search so far.",
            "- Mixed-bit allocation beats pure 4-bit everywhere and beats the earlier fixed group-size sweep.",
            "- Global allocation beats attention-only or MLP-only variants; the budget wants to be shared across the network.",
            "- Tiered 4->6-bit allocation is now slightly ahead of flat 5-bit allocation.",
            "- Folded MLP permutations are a dead end in this setup; they blow up quick perplexity without helping size.",
        ]
    )

    if runner_state:
        lines.extend(
            [
                "",
                "## Live Run",
                "",
                f"- Completed experiments in active run: {runner_state.get('completed_experiments', 0)}",
                f"- Active-run best: {runner_state.get('best_full_perplexity')}",
                f"- Active-run best path: {runner_state.get('best_run')}",
                f"- Last completed run: {runner_state.get('last_run')}",
                f"- Last status: {runner_state.get('last_status')}",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    results_tsv = Path(args.results_tsv).expanduser().resolve()
    campaign_path = Path(args.campaign).expanduser().resolve()
    runner_state_path = Path(args.runner_state).expanduser().resolve()
    perplexity_json_path = Path(args.perplexity_json).expanduser().resolve()
    output_svg = Path(args.output_svg).expanduser().resolve()
    output_md = Path(args.output_md).expanduser().resolve()

    rows = load_rows(results_tsv)
    campaign = load_campaign_metadata(campaign_path)
    runner_state = load_runner_state(runner_state_path)
    original_ref = load_original_reference(perplexity_json_path)

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    render_svg(rows, campaign, runner_state, original_ref, output_svg)
    output_md.write_text(summarize(rows, campaign, runner_state, original_ref), encoding="utf-8")
    print(f"wrote {output_svg}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
