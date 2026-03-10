#!/usr/bin/env python3
"""Basic automatic sanity checks for AQ1 inference logs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check AQ1 inference output for obvious degeneration")
    parser.add_argument("--log", required=True, help="Inference log file")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--min-chars", type=int, default=24, help="Minimum response length")
    parser.add_argument("--max-token-run", type=int, default=8, help="Maximum repeated token run")
    parser.add_argument("--max-4gram-count", type=int, default=3, help="Maximum count for a repeated 4-gram")
    return parser.parse_args()


def normalize(text: str) -> str:
    text = text.replace("<|im_end|>", " ")
    text = text.replace("<think>", " ")
    text = text.replace("</think>", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def longest_token_run(tokens: list[str]) -> int:
    if not tokens:
        return 0
    best = 1
    current = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def max_ngram_count(tokens: list[str], n: int) -> int:
    if len(tokens) < n:
        return 0
    counts = Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return max(counts.values()) if counts else 0


def extract_responses(log_path: Path) -> list[str]:
    responses: list[str] = []
    pattern = re.compile(r"^(Response|Assistant):\s*(.*)$")
    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                responses.append(match.group(2))
    return responses


def main() -> int:
    args = parse_args()
    log_path = Path(args.log).expanduser().resolve()
    if not log_path.exists():
        print(f"missing log: {log_path}", file=sys.stderr)
        return 2

    raw_responses = extract_responses(log_path)
    checks = []
    overall_ok = True

    for response in raw_responses:
        cleaned = normalize(response)
        tokens = cleaned.split()
        token_run = longest_token_run(tokens)
        fourgram_count = max_ngram_count(tokens, 4)
        ok = True
        reasons: list[str] = []

        if len(cleaned) < args.min_chars:
            ok = False
            reasons.append("too_short")
        if token_run > args.max_token_run:
            ok = False
            reasons.append("token_run")
        if fourgram_count > args.max_4gram_count:
            ok = False
            reasons.append("repeat_4gram")

        checks.append(
            {
                "response": cleaned,
                "chars": len(cleaned),
                "tokens": len(tokens),
                "longest_token_run": token_run,
                "max_4gram_count": fourgram_count,
                "ok": ok,
                "reasons": reasons,
            }
        )
        overall_ok = overall_ok and ok

    if not raw_responses:
        overall_ok = False

    result = {
        "log": str(log_path),
        "response_count": len(raw_responses),
        "ok": overall_ok,
        "checks": checks,
    }

    if args.format == "json":
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"log:            {result['log']}")
        print(f"response_count: {result['response_count']}")
        print(f"ok:             {result['ok']}")
        for idx, check in enumerate(checks, start=1):
            reasons = ",".join(check["reasons"]) if check["reasons"] else "-"
            print(
                f"response_{idx}: ok={check['ok']} chars={check['chars']} "
                f"tokens={check['tokens']} token_run={check['longest_token_run']} "
                f"repeat4={check['max_4gram_count']} reasons={reasons}"
            )

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
