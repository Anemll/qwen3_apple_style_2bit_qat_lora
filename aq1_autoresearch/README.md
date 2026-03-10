# AQ1 Autoresearch

This folder adapts the `autoresearch` pattern to AQ1 / ANEMLL-QUANT-1.

The default loop here is not training-first. It is initialization-first:

- generate a quantized candidate checkpoint
- measure perplexity on a fixed evaluation setup
- optionally run FP16 snap and inference sanity
- keep or discard the change

For Apple Neural Engine work, this is the right default. Better initialization and better LUT construction often matter before any extra recovery training.

## Goal

For a fixed campaign, improve AQ1 quantization for Apple Silicon and future ANE deployment by improving:

- initial quantized checkpoint quality
- full perplexity on a fixed dataset
- projected deployed payload under a fixed size budget
- FP16 snap readiness
- deterministic post-snap inference stability

The main score is perplexity. Lower is better.

## Score Contract

Default score order:

1. full perplexity from `scripts/measure_perplexity.py`
2. quick perplexity from `scripts/init_model_v2.py --ppl` or `scripts/quick_perplexity.py`
3. hard size gate when the campaign enables a payload budget
4. optional gates: `snap_ok` and `inference_ok`
5. tie-breakers: simpler code, smaller search space, lower wall time

Use `aq1_autoresearch/score_run.py` to summarize a candidate directory. It understands:

- initialization artifacts from `scripts/init_model_v2.py`
- training artifacts with `loss.csv`
- full PPL from `perplexity.log` or `results/perplexity.json`
- quick PPL embedded in `init_metrics.json`
- projected payload size from the quantized checkpoint itself

## Projected Size Contract

Do not use raw `.pt` file size as the budget metric. AQ1 checkpoints often retain dense helper tensors such as `_Q`, which makes the checkpoint far larger than the intended deployed representation.

The default projected-size estimate in `aq1_autoresearch/score_run.py` assumes:

- quantized weights are stored as packed LUT indices at each layer's actual bitwidth
- `scale_A`, `scale_B`, `rank_magnitude`, and `lut` are stored at FP16
- the size gate applies to deployed payload, not to scratch artifacts
- `score_run.py` is executed in an environment where `torch` can load the checkpoint

Per-layer average bits are therefore:

`avg_bits_per_weight = (index_bits + scale_bits + lut_bits) / quantized_weight_count`

This means mixed-bit experiments are allowed, but they must stay within the campaign ceiling. It also means folded transforms such as offline SpinQuant-style rotations are size-neutral unless they add runtime tensors. If a method requires explicit stored rotations or other extra metadata at inference time, count that payload too or reject the candidate.

## Repo Primitives

The core init/perplexity commands already exist in this repo:

- initialization: `scripts/init_model_v2.py`
- quick PPL estimate: `scripts/quick_perplexity.py`
- full PPL: `scripts/measure_perplexity.py`
- per-layer LUT search: `scripts/select_best_lut_per_layer.py`
- global LUT family sweep: `scripts/apply_lut_candidates.py`
- rank/filter screening for LUT candidates: `scripts/eval_lut_candidates.py`
- LUT baking: `scripts/bake_lut.py`
- FP16 snap proxy: `scripts/snap_and_test_v2.py`
- deterministic inference sanity: `scripts/test_inference.py`

Training is still available, but it should be treated as phase two, not the baseline loop:

- recovery / KD training: `scripts/train_v2_simple.py`

## Good First Campaign Boundaries

Keep these fixed inside one campaign:

- `model_id`
- eval dataset or text file
- PPL mode and chunk count
- device and dtype
- quant config family
- fixed sanity prompts

Change one family of variables at a time:

- block-scale / SVD initialization
- `group_size` or per-layer group search
- LUT family design
- LUT `max_abs` heuristics
- per-layer LUT selection metric
- activation-aware or AWQ-like weighting
- mixed-bit or mixed-LUT allocation under a fixed budget
- SpinQuant-style folded rotations or similar pre-quantization transforms
- MLP-only vs attention-inclusive initialization
- FP16-safe LUT repair and snap behavior

Avoid mixing initialization changes, recovery training, and downstream exporter changes in one experiment.

## Recommended Verification Ladder

Use the cheapest reliable gate first:

1. candidate checkpoint builds and loads
2. quick PPL is not obviously worse
3. full PPL improves or is tied
4. FP16 snap succeeds if the campaign requires deployment readiness
5. post-snap inference sanity succeeds if the campaign requires deployment readiness

For early init work, quick PPL is a screening metric only. Keep decisions should be based on full PPL whenever possible.

## Retention Policy

Disk usage will balloon if every experiment keeps all `.pt` artifacts and LUT candidate trees.

After an experiment is scored and logged:

- discard runs: keep only metadata and logs, delete heavy artifacts
- keep runs: retain at most one checkpoint by default, plus metadata and logs
- if you want maximum disk savings, prune kept runs too and rely on commit + campaign + score snapshot for reproducibility

Use `aq1_autoresearch/cleanup_run.py` after each experiment. It writes:

- `score.json`
- `cleanup_manifest.json`
- `campaign_snapshot.md`

and then prunes heavy artifacts such as:

- `.pt`, `.pth`, `.bin`, `.safetensors`, `.npz`, `.npy`
- named bulky directories such as `lut_candidates`

## Suggested Keep / Discard Rule

Default keep rule:

- keep if full perplexity improves by a meaningful amount on the fixed evaluation setup
- keep only if projected payload stays within the campaign size budget
- keep smaller gains only if the change is clearly simpler or more ANE-friendly
- discard if full PPL regresses
- discard if snap or inference fails when those gates are enabled for the campaign

## Candidate Families Worth Exploring

Strong first ideas:

- new LUT families beyond `uniform` and `fp4_dense`
- per-layer LUT selection using `weighted_mse` vs activation-aware weighting
- AWQ-like initialization using activation-aware importance for LUT or scale selection
- improved `max_abs` and percentile heuristics for LUT construction
- per-layer or searched `group_size`
- better block-scale statistics before SVD
- tighter FP16-safe LUT snapping and duplicate repair

More invasive ideas:

- different scale factorization or normalization rules in V2 init
- layer-type-specific LUT/search policies
- optional calibration passes for activation-aware selection
- a second-stage recovery fine-tune after the best initialization candidates are identified

## Recommended Campaign Flow

1. Copy `aq1_autoresearch/campaign_template.md` to `aq1_autoresearch/campaign.md`.
2. Create a fresh branch such as `aq1-auto/<tag>`.
3. Run one baseline initialization candidate.
4. Measure full PPL and log it in `aq1_autoresearch/results.tsv`.
5. Let the agent iterate one coherent change family at a time.
6. For every candidate:
   - generate checkpoint
   - measure PPL
   - optionally snap
   - optionally run inference sanity
   - score
   - log
   - keep or discard
   - prune heavy artifacts

## Folder Contents

- `README.md`: overview and rules
- `program.md`: direct instructions for a coding agent
- `campaign_template.md`: fields to fill before a campaign
- `results.tsv`: experiment ledger
- `score_run.py`: summarize candidate metrics
- `cleanup_run.py`: prune heavy artifacts after logging
- `check_inference.py`: repetition / emptiness gate for inference logs
