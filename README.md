# ctxcomp — context compression for long-horizon LLM agents

A standalone harness for comparing context-compression policies on **AppWorld**.

It exists to answer one question with numbers: *when an agent's growing history is
replaced by a bounded summary, what does that cost — in success rate, in
multi-run reliability, and in the context/step budget?*

## Methods

| # | method | kind | what it is |
|---|---|---|---|
| 1 | `no_compression` | none | full history, upper bound |
| 2 | `fifo` | history | recency window, keep last k turns |
| 3 | `llmlingua` | history | token-level pruning |
| 4 | `prompting_o` | history | OpenClaw two-stage checkpoint prompt |
| 5 | `prompting_h` | history | Hermes structured checkpoint + reference-only prefix |
| 6 | `trace` | history | verifier-optimised compression template |
| 7 | `acon_ut` | history | ACON utility-optimised guideline |
| 8 | `acon_utco` | history | ACON utility+compression-optimised guideline |

`hist_*` compresses the **history**; the harness also supports `obs_*`, which
compresses **observations** only. `no_compression` runs no policy at all.

Provenance, upstream commits and licence notes for every prompt are in
`docs/METHODS.md` and `docs/CREDITS.md`.

## Metrics

Reported per method, matching what the two source papers print so the numbers are
directly comparable:

| metric | definition |
|---|---|
| **Avg Acc** | mean task success over `repeats` independent runs |
| **Pass²** | fraction solved in **all** runs — multi-run reliability |
| **Pass@2** | fraction solved in **at least one** run — task coverage |
| **Easy / Medium / Hard** | per-difficulty success (57 / 48 / 63 tasks) |
| **Steps** | mean environment interactions per task |
| **Peak** | max input context length over all steps, **including** the system prompt, in 10³ tokens |
| **Dep.** | `Σ_t ((n_i + 2·n_o)·n_o)/2` with `n_i` **excluding** the system prompt, in 10⁶ |

`Acc`/`Peak`/`Dep` follow the ACON paper's Table 1; `Pass²`/`Pass@2` follow TRACE's
Table 1. Both are computed here from the trajectory rather than trusted from a log
line, so they can be recomputed for any past run.

## Usage

```bash
python scripts/make_configs.py                 # methods.yml -> ctxcomp/configs/*
bash   scripts/run_suite.sh --stamp my_run     # run every method
python scripts/compute_table.py --stamp my_run # the table above
```

`--repeats 2` is needed for Pass²/Pass@2; the paper's difficulty split needs only
`--repeats 1`, so a first pass is cheap and the reliability columns can be filled
by a second pass later (runs are stored per repeat and joined afterwards).

## Layout

```
methods.yml              the single source of truth: one entry per method
ctxcomp/
  methods/               one CompressionPolicy per method
  prompts/               prompt templates, one directory per prompt-driven method
  metrics/tokens.py      Steps / Peak / Dep
  metrics/table.py       the table above
  runner/                AppWorld execution adapter
  configs/               generated per-method configs (do not edit by hand)
scripts/                 make_configs.py, run_suite.sh, compute_table.py
docs/                    methods, provenance, credits, experiment log
```

## Status

The eight baseline policies are implemented. The project's own contribution (a
rate-distortion, rollout-free training objective for the compressor) is designed
in `docs/DESIGN.md` and **not yet implemented** — `ctxcomp/methods/` is where it
lands, behind the same `CompressionPolicy` interface the baselines use.
