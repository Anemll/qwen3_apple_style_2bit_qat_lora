# AQ1 Campaign

## Identity

- tag: `qwen06b-init-ppl`
- stable branch: `aq1-auto/qwen06b-init-ppl`
- output root: `runs/aq1_auto/qwen06b-init-ppl`

## Fixed Inputs

- model id: `Qwen/Qwen3-0.6B`
- device: `mps`
- dtype for PPL: `fp16`
- config preset: `q4a4`
- eval dataset: `wikitext2`
- full PPL chunks: `20`
- quick PPL chunks: `14`

## Size Budget

- size metric: projected deployed payload from `aq1_autoresearch/score_run.py`
- size estimate mode: packed LUT indices + FP16 scales/LUTs
- baseline projected payload: `248.52 MiB`
- baseline average bits per weight: `4.7337`
- max payload growth vs baseline: `5%`
- size ceiling: `260.94 MiB`
- average bits per weight ceiling: `4.9704`

Notes:

- raw `.pt` file size is not the metric because checkpoints still store dense helper tensors like `_Q`
- folded changes such as LUT search, `group_size`, or offline SpinQuant-style rotations are size-neutral unless they add runtime tensors
- explicit runtime rotations or extra per-layer metadata must be counted against this budget

## Primary Candidate Type

- baseline mode: `init_v2`

## Baseline Command

```bash
python scripts/init_model_v2.py \
  --output runs/aq1_auto/qwen06b-init-ppl/baseline \
  --config q4a4 \
  --group-size 32 \
  --search-lut \
  --ppl \
  --ppl-chunks 14
```

## Full PPL Command

```bash
python scripts/measure_perplexity.py runs/aq1_auto/qwen06b-init-ppl/baseline/v2_tightened.pt \
  --config runs/aq1_auto/qwen06b-init-ppl/baseline/config.json \
  --device mps \
  --dtype fp16 \
  --max-chunks 20 \
  > runs/aq1_auto/qwen06b-init-ppl/baseline/perplexity.log 2>&1
```

If `v2_tightened.pt` is not present, use `v2_initial.pt`.

## Optional Deployment-Readiness Gates

- require snap gate: `yes`
- require inference gate: `no`

Example snap command:

```bash
python scripts/snap_and_test_v2.py \
  --checkpoint runs/aq1_auto/qwen06b-init-ppl/baseline/v2_tightened.pt \
  --fp16 \
  --no-test \
  --output runs/aq1_auto/qwen06b-init-ppl/baseline/snapped_fp16.pt
```

## Retention Policy

- discard runs: prune all heavy artifacts after logging
- keep runs: retain at most one winning checkpoint by default

Example cleanup commands:

```bash
python aq1_autoresearch/cleanup_run.py \
  --run-dir runs/aq1_auto/qwen06b-init-ppl/baseline \
  --status keep

python aq1_autoresearch/cleanup_run.py \
  --run-dir runs/aq1_auto/qwen06b-init-ppl/exp_001 \
  --status discard \
  --retain-checkpoint none
```

## Fixed Sanity Prompts

1. `Explain why low-rank scales help quantization.`
2. `What is the tradeoff between LUT size and model quality?`
3. `Why can FP16 snap break a quantized checkpoint if magnitudes are unstable?`

## Candidate Families To Explore

- LUT families and LUT shape
- activation-aware or AWQ-like weighting
- mixed-bit or mixed-LUT allocation under the size budget
- per-layer LUT selection metric
- `group_size` or per-layer group search
- SpinQuant-style folded rotations or similar pre-quantization transforms
- SVD or block-scale initialization details
- FP16-safe LUT repair / snap behavior

## Keep Rule

- primary metric: lower full perplexity
- quick PPL is only a screen
- projected deployed payload must stay within the size budget
- require snap to pass
- keep ties only when the change is clearly simpler or more ANE-friendly
