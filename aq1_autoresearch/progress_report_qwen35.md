# AQ1 ANE-Native Quantization Summary

- Model id: Qwen/Qwen3.5-0.8B
- Quant target: q4a4
- Baseline / ceiling bits: unknown / unknown bpw
- Completed rows logged: 1
- Full perplexity evaluations: 1
- Quick-screen-only rows: 0
- Baseline full perplexity: 17.83
- Best full perplexity: 17.83 (qwen35_0p8b_init)
- Improvement vs baseline: 0.00 (0.00%)
- Best payload / avg bits: n/a MiB / n/a bits per weight
- Original model perplexity: 16.54 (Qwen/Qwen3.5-0.8B, full PPL)
- Quantized baseline delta vs original: +1.29
- Best run delta vs original: +1.29
- Recovered quantization gap: 0.00 / 1.29 (0.00%)

## New Best Sequence

- #00 qwen35_0p8b_init: 17.83 [init_v2 q4a4 group_size=16]

## Best By Family

- baseline: 17.83 (qwen35_0p8b_init)

## Best Overall Runs

- 17.83 | qwen35_0p8b_init | init_v2 q4a4 group_size=16 | n/a MiB | n/a bits | delta q4 +0.00 | delta orig +1.29

## Current Read

- Initial baseline only; no comparative AQ1 candidates yet.
