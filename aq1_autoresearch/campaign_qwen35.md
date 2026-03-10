# AQ1 Campaign

## Identity

- tag: `qwen35-init-ppl`
- stable branch: `aq1-auto/qwen35-0p8b-check`
- output root: `runs/qwen35_0p8b_init`

## Fixed Inputs

- model id: `Qwen/Qwen3.5-0.8B`
- device: `mps`
- dtype for PPL: `fp16`
- config preset: `q4a4`
- eval dataset: `wikitext2`
- full PPL chunks: `20`
- quick PPL chunks: `14`

## Size Budget

- size metric: projected deployed payload from `aq1_autoresearch/score_run.py`
- size estimate mode: packed LUT indices + FP16 scales/LUTs
- baseline projected payload: `unknown`
- baseline average bits per weight: `unknown`
- max payload growth vs baseline: `5%`

## Primary Candidate Type

- baseline mode: `init_v2`
