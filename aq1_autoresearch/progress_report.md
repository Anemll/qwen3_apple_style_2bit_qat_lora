# AQ1 ANE-Native Quantization Summary

- Model id: Qwen/Qwen3-0.6B
- Quant target: q4a4
- Baseline / ceiling bits: 4.7337 / 4.9704 bpw
- Completed rows logged: 26
- Full perplexity evaluations: 18
- Quick-screen-only rows: 8
- Baseline full perplexity: 32.86
- Best full perplexity: 28.03 (exp_001_tier_all_eff_b6_g16_p5p0)
- Improvement vs baseline: 4.83 (14.70%)
- Best payload / avg bits: 260.90 MiB / 4.9694 bits per weight
- Original model perplexity: 22.38 (Qwen/Qwen3-0.6B, CPU fp32, same max_chunks=20)
- Quantized baseline delta vs original: +10.48
- Best run delta vs original: +5.65
- Recovered quantization gap: 4.83 / 10.48 (46.09%)

## New Best Sequence

- #00 baseline: 32.86 [init_v2 q4a4 group_size=32 search_lut fp4_dense-all]
- #01 exp_001: 29.72 [group_size=16 with search_lut]
- #04 exp_001_mix_attn_eff_b5_g16: 28.15 [attention-only 5-bit upgrades ranked by local improvement per extra bit]
- #05 exp_002_mix_all_eff_b5_g16: 28.09 [global 5-bit upgrades ranked by local improvement per extra bit]
- #16 exp_001_tier_all_eff_b6_g16_p5p0: 28.03 [all tiered 4->6-bit upgrades via greedy efficiency at group_size=16 within a 5.00% payload budget]

## Best By Family

- mixedbit: 28.03 (exp_001_tier_all_eff_b6_g16_p5p0)
- group_size: 29.72 (exp_001)
- group_search: 30.26 (exp_003)
- baseline: 32.86 (baseline)

## Best Overall Runs

- 28.03 | exp_001_tier_all_eff_b6_g16_p5p0 | all tiered 4->6-bit upgrades via greedy efficiency at group_size=16 within a 5.00% payload budget | 260.90 MiB | 4.9694 bits | delta q4 -4.83 | delta orig +5.65
- 28.09 | exp_002_mix_all_eff_b5_g16 | global 5-bit upgrades ranked by local improvement per extra bit | 260.90 MiB | 4.9694 bits | delta q4 -4.77 | delta orig +5.71
- 28.10 | exp_004_mix_all_eff_b5_g16_p3p5 | all 5-bit upgrades ranked by local improvement per extra bit at group_size=16 within a 3.50% payload budget | 257.02 MiB | 4.8956 bits | delta q4 -4.76 | delta orig +5.72
- 28.10 | exp_007_mix_attn_eff_b5_g16_p3p5 | attn 5-bit upgrades ranked by local improvement per extra bit at group_size=16 within a 3.50% payload budget | 257.02 MiB | 4.8956 bits | delta q4 -4.76 | delta orig +5.72
- 28.12 | exp_003_tier_all_eff_b6_g16_p3p5 | all tiered 4->6-bit upgrades via greedy efficiency at group_size=16 within a 3.50% payload budget | 257.14 MiB | 4.8980 bits | delta q4 -4.74 | delta orig +5.74
- 28.15 | exp_001_mix_attn_eff_b5_g16 | attention-only 5-bit upgrades ranked by local improvement per extra bit | 260.77 MiB | 4.9671 bits | delta q4 -4.71 | delta orig +5.77

## Current Read

- g16 is clearly better than g32 in this search so far.
- Mixed-bit allocation beats pure 4-bit everywhere and beats the earlier fixed group-size sweep.
- Global allocation beats attention-only or MLP-only variants; the budget wants to be shared across the network.
- Tiered 4->6-bit allocation is now slightly ahead of flat 5-bit allocation.
- Folded MLP permutations are a dead end in this setup; they blow up quick perplexity without helping size.

## Live Run

- Completed experiments in active run: 10
- Active-run best: 28.03
- Active-run best path: /Users/anemll/SourceRelease/GITHUB/ML_playground/aq1/runs/aq1_auto/qwen06b-init-ppl/auto6h_20260310_run8/exp_001_tier_all_eff_b6_g16_p5p0
- Last completed run: /Users/anemll/SourceRelease/GITHUB/ML_playground/aq1/runs/aq1_auto/qwen06b-init-ppl/auto6h_20260310_run8/exp_010_mix_all_eff_b5_g8_p5p0
- Last status: discard
