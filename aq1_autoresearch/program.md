# AQ1 Autoresearch Program

This program turns AQ1 quantization work into an `autoresearch`-style loop.

You are not optimizing generic training loss. You are optimizing initialization quality for quantized checkpoints that may later ship to Apple Neural Engine style deployment.

The correct mindset is:

- improve perplexity first
- use quick PPL only as a fast screen
- keep FP16 snap compatibility in view
- prefer simpler ANE-friendly mechanisms over brittle tricks

## Read First

Before doing anything else, read:

- `aq1_autoresearch/README.md`
- `aq1_autoresearch/campaign.md` if it exists
- `AQ1.md`
- `scripts/init_model_v2.py`
- `scripts/measure_perplexity.py`
- `scripts/quick_perplexity.py`
- `scripts/select_best_lut_per_layer.py`

If `aq1_autoresearch/campaign.md` does not exist, stop and ask the human to create it from `campaign_template.md`.

## Objective

For a fixed campaign:

- improve full perplexity on the fixed eval setup
- use quick perplexity as a screening metric only
- stay within the campaign's projected payload budget
- optionally require `snap_ok = true`
- optionally require `inference_ok = true`

Primary metric:

- full perplexity from `scripts/measure_perplexity.py`

Secondary metric:

- quick perplexity from `init_model_v2.py --ppl` or `quick_perplexity.py`

Hard gates when enabled by the campaign:

- projected deployed payload must stay within budget
- FP16 snap must succeed
- deterministic inference must pass sanity

## In Scope

Allowed change areas:

- `scripts/init_model_v2.py`
- `scripts/quick_perplexity.py`
- `scripts/measure_perplexity.py`
- `scripts/select_best_lut_per_layer.py`
- `scripts/apply_lut_candidates.py`
- `scripts/eval_lut_candidates.py`
- `scripts/bake_lut.py`
- `qat_lora/ane_qat_linear_v2.py`
- small helper code directly tied to initialization, PPL measurement, or deployment readiness

## Out of Scope By Default

Do not change these unless the human explicitly opens a new campaign:

- model family
- eval dataset or text file
- fixed prompt set used for sanity checks
- downstream exporter outside this repo
- TPU-based recovery training
- LoRA recovery stage

For the first campaign, do not mix initialization improvements with unrelated SFT or recovery training behavior.

## Safe Git Workflow

Use a stable campaign branch and disposable scratch branches.

- Stable branch: `aq1-auto/<tag>`
- Scratch branch pattern: `aq1-exp/<tag>-NN`

For each experiment:

1. Start from the latest kept commit on the stable campaign branch.
2. Create a new scratch branch from that kept commit.
3. Make one coherent change family.
4. Commit the code change on the scratch branch.
5. Run the candidate.
6. Log the result.
7. If kept, fast-forward merge the scratch branch into the stable branch.
8. If discarded, switch back to the stable branch and delete the scratch branch.

Avoid destructive history rewriting for the campaign branch.

## Baseline Setup

Before autonomous iteration:

1. Verify all campaign paths exist.
2. Verify the fixed eval setup is available.
3. Run the baseline command from `campaign.md`.
4. If the baseline command only produced quick PPL, run full PPL and capture the log:

```bash
python scripts/measure_perplexity.py <checkpoint> \
  --config <config-or-preset> \
  --device <device> \
  --dtype <dtype> \
  --max-chunks <N> \
  > <run_dir>/perplexity.log 2>&1
```

5. If the campaign requires deployment-readiness gates, run snap:

```bash
python scripts/snap_and_test_v2.py \
  --checkpoint <checkpoint> \
  --fp16 \
  --no-test \
  --output <run_dir>/snapped_fp16.pt \
  > <run_dir>/snap.log 2>&1
```

6. If the campaign requires inference sanity, run deterministic prompts and append output to one log:

```bash
python scripts/test_inference.py <run_dir>/snapped_fp16.pt \
  --prompt "Explain why low-rank scales help quantization." \
  --max-tokens 96 \
  --no-thinking \
  > <run_dir>/inference.log 2>&1
```

Repeat the inference command for the other fixed prompts from `campaign.md`, appending to the same log.

7. Score the baseline:

```bash
python aq1_autoresearch/score_run.py \
  --run-dir <run_dir> \
  --baseline-run-dir <baseline_run_dir> \
  --max-size-growth-pct <pct>
```

Run this in the repo venv or another environment with `torch` installed when the size budget is enabled.

8. Append the baseline row to `aq1_autoresearch/results.tsv`.
9. Prune heavy artifacts once the baseline has been logged:

```bash
python aq1_autoresearch/cleanup_run.py \
  --run-dir <run_dir> \
  --status keep
```

## Per-Experiment Loop

For every experiment, do exactly this:

1. Create a new scratch branch from the latest kept commit.
2. Make one coherent change.
3. Commit the change.
4. Generate one candidate checkpoint.
5. Measure full PPL.
6. Optionally snap.
7. Optionally run deterministic inference sanity.
8. Score the candidate.
9. Log the result in `aq1_autoresearch/results.tsv`.
10. Decide keep or discard.
11. Prune heavy artifacts with the appropriate retention mode.

## Candidate Shapes

Prefer the cheapest valid experiment shape for the change you made.

### Shape A: Direct init candidate

Use when changing initialization logic in `init_model_v2.py` or `ane_qat_linear_v2.py`.

```bash
python scripts/init_model_v2.py \
  --output <run_dir> \
  --config <preset> \
  --group-size <group_size> \
  --ppl \
  --ppl-chunks <quick_chunks> \
  > <run_dir>/init.log 2>&1
```

Then run full PPL on the best checkpoint for this candidate, usually `v2_tightened.pt` if present, otherwise `v2_initial.pt`.

### Shape B: Per-layer LUT hybrid

Use when changing candidate LUT families or selection logic.

```bash
python scripts/select_best_lut_per_layer.py \
  --from-scratch \
  -o <run_dir>/hybrid.pt \
  --output-stats <run_dir>/lut_stats.json \
  --families <families> \
  --metric <metric> \
  > <run_dir>/select.log 2>&1
```

Then run full PPL on `<run_dir>/hybrid.pt`.

### Shape C: Global LUT family sweep

Use when comparing entire LUT families before doing per-layer hybrids.

```bash
python scripts/apply_lut_candidates.py <checkpoint> \
  -o <run_dir>/lut_candidates \
  > <run_dir>/apply.log 2>&1

python scripts/eval_lut_candidates.py <run_dir>/lut_candidates \
  --max-chunks <quick_chunks> \
  > <run_dir>/eval.log 2>&1
```

Only promote the best candidates to full PPL runs.

## Scoring

Use:

```bash
python aq1_autoresearch/score_run.py \
  --run-dir <run_dir> \
  --baseline-run-dir <baseline_run_dir> \
  --max-size-growth-pct <pct> \
  --format json
```

Read:

- `score`
- `full_perplexity`
- `quick_perplexity`
- `projected_payload_mib`
- `avg_bits_per_weight`
- `size_ok`
- `best_eval_loss`
- `final_eval_loss`
- `elapsed_sec`
- `candidate_checkpoint`

Interpretation:

- for init-first campaigns, `score` should usually be full PPL
- if only quick PPL exists, the result is provisional
- if neither PPL metric exists, the run is not ready to compare

## Cleanup

After the result is logged, prune heavy artifacts:

Discarded run:

```bash
python aq1_autoresearch/cleanup_run.py \
  --run-dir <run_dir> \
  --status discard \
  --retain-checkpoint none
```

Kept run:

```bash
python aq1_autoresearch/cleanup_run.py \
  --run-dir <run_dir> \
  --status keep
```

This keeps the run reproducible by preserving metadata files and deleting bulky checkpoints and known heavy directories.

## Keep Rules

Keep a candidate when:

- full perplexity improves on the fixed setup, and
- required gates pass

Also keep when:

- score is effectively tied, but the change is clearly simpler, or
- score is effectively tied, but the change is more FP16-safe or more ANE-friendly

Discard when:

- checkpoint generation crashes
- full PPL regresses
- projected payload exceeds the campaign budget
- required gates fail

## Candidate Ideas

Strong first ideas:

- change LUT families beyond `uniform` and `fp4_dense`
- test activation-aware or AWQ-like selection heuristics
- test mixed-bit or mixed-LUT allocation under the fixed size budget
- compare `weighted_mse` vs `activation_mse`
- tune `group_size` or per-layer group search
- test SpinQuant-style folded rotations or similar pre-quantization transforms
- improve `max_abs` and percentile heuristics
- improve block-scale statistics before SVD
- improve FP16 duplicate repair and LUT snapping
- split policies by MLP vs attention

Second-stage ideas after good init candidates exist:

- short recovery training runs
- Q4 to Q2 progressive conversion after the best initialization scheme is identified
- deployment-specific gates tied to downstream CoreML export

## Logging

Append one row per experiment to `aq1_autoresearch/results.tsv`.

Columns:

- `commit`
- `run_dir`
- `artifact_type`
- `candidate_checkpoint`
- `score`
- `full_perplexity`
- `quick_perplexity`
- `projected_payload_mib`
- `avg_bits_per_weight`
- `size_ok`
- `best_eval_loss`
- `final_eval_loss`
- `elapsed_sec`
- `snap_ok`
- `inference_ok`
- `status`
- `retention`
- `change_family`
- `description`

Status values:

- `keep`
- `discard`
- `crash`

## Stopping Rule

If you are running autonomously, keep going until:

- the human interrupts you
- the campaign budget is exhausted
- you can no longer produce valid candidates that improve full PPL

If you feel stuck, simplify and go back to one-change init experiments.
