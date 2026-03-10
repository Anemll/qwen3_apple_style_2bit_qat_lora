# AQ1 Campaign Template

Copy this file to `aq1_autoresearch/campaign.md` and fill in the real values before starting a loop.

## Identity

- tag: `replace-me`
- stable branch: `aq1-auto/<tag>`
- output root: `runs/aq1_auto/<tag>`

## Fixed Inputs

- model id: `Qwen/Qwen3-0.6B`
- device: `mps`
- dtype for PPL: `fp16`
- config preset: `q4a4`
- eval dataset: `wikitext2`
- full PPL chunks: `20`
- quick PPL chunks: `14`

## Size Budget

Fill this in after the baseline is scored.

- size metric: projected deployed payload from `aq1_autoresearch/score_run.py`
- size estimate mode: packed LUT indices + FP16 scales/LUTs
- baseline projected payload: `fill after baseline`
- baseline average bits per weight: `fill after baseline`
- max payload growth vs baseline: `5%`
- size ceiling: `derive from baseline`
- average bits per weight ceiling: `derive from baseline`

Notes:

- raw `.pt` size is not the budget metric
- folded changes such as LUT search, `group_size`, or offline SpinQuant-style rotations are size-neutral unless they add runtime tensors
- explicit runtime rotations or extra per-layer metadata must be counted against this budget

## Primary Candidate Type

Choose one as the baseline loop:

- `init_v2`
- `per_layer_lut_hybrid`
- `global_lut_sweep`

Example:

- baseline mode: `init_v2`

## Baseline Command

Fill in one command and keep it fixed for the whole campaign unless you explicitly start a new campaign.

Example init-first baseline:

```bash
python scripts/init_model_v2.py \
  --output runs/aq1_auto/<tag>/baseline \
  --config q4a4 \
  --group-size 32 \
  --search-lut \
  --ppl \
  --ppl-chunks 14
```

Example per-layer LUT baseline:

```bash
python scripts/select_best_lut_per_layer.py \
  --from-scratch \
  -o runs/aq1_auto/<tag>/baseline/hybrid.pt \
  --output-stats runs/aq1_auto/<tag>/baseline/lut_stats.json \
  --families E,F,G,A,B,C,D \
  --metric weighted_mse
```

## Full PPL Command

Use the same PPL setup for every candidate in the campaign.

```bash
python scripts/measure_perplexity.py runs/aq1_auto/<tag>/baseline/v2_tightened.pt \
  --config runs/aq1_auto/<tag>/baseline/config.json \
  --device mps \
  --dtype fp16 \
  --max-chunks 20 \
  > runs/aq1_auto/<tag>/baseline/perplexity.log 2>&1
```

If the candidate is not an `init_model_v2.py` directory, point the checkpoint to the produced `.pt` file and use the correct `--config` preset or config path.

## Optional Deployment-Readiness Gates

Decide whether these are required in this campaign.

- require snap gate: `yes/no`
- require inference gate: `yes/no`

Example snap command:

```bash
python scripts/snap_and_test_v2.py \
  --checkpoint runs/aq1_auto/<tag>/baseline/v2_tightened.pt \
  --fp16 \
  --no-test \
  --output runs/aq1_auto/<tag>/baseline/snapped_fp16.pt
```

## Retention Policy

- discard runs: prune all heavy artifacts after logging
- keep runs: retain at most one winning checkpoint unless you explicitly choose `none`

Example cleanup commands:

```bash
python aq1_autoresearch/cleanup_run.py \
  --run-dir runs/aq1_auto/<tag>/baseline \
  --status keep

python aq1_autoresearch/cleanup_run.py \
  --run-dir runs/aq1_auto/<tag>/exp_001 \
  --status discard \
  --retain-checkpoint none
```

## Fixed Sanity Prompts

Use the same prompts for every experiment when inference gating is enabled.

1. `Explain why low-rank scales help quantization.`
2. `What is the tradeoff between LUT size and model quality?`
3. `Why can FP16 snap break a quantized checkpoint if magnitudes are unstable?`

## Candidate Families To Explore

Pick one family per experiment:

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
- require enabled gates to pass
- keep ties only when the change is clearly simpler or more ANE-friendly

## Notes

- Keep eval data, prompts, device, and dtype fixed.
- Leave TPU-based training out of the first campaign unless the goal explicitly includes recovery training.
- If you later add downstream CoreML / ANE export validation, record that command here and make it a required gate.
