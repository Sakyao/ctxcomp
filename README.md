# ctxcomp — context compression for long-horizon LLM agents

A standalone harness for comparing context-compression policies on **AppWorld**.

It exists to answer one question with numbers: *when an agent's growing history is
replaced by a bounded summary, what does that cost — in success rate, in
multi-run reliability, and in the context/step budget?*

## Two axes

ACON compresses two different things and reports them as two separate AppWorld
lines (its section 4.2 has one subsection per axis):

```
history      h'_t = f(h_t; P_hist)          if |h_t| > T_hist      T_hist = 4096
observation  o'_t = f(o_t, h_{t-1}; P_obs)  if |o_t| > T_obs       T_obs  = 1024
```

They differ in input, in output shape, in threshold, and in the prompt that gets
optimised: `P_hist` and `P_obs` are learned separately. So every method except the
uncompressed reference has two rows, and comparing a history row against an
observation row is a category error.

Note what the observation axis is *not*: it is not "drop old observations". By
equation (4) it compresses the **current** observation when that observation alone
is too long, conditioned on the history so far.

## Methods

Fifteen rows: one uncompressed reference, and seven methods on each axis.

| method | history | observation | mechanism |
|---|---|---|---|
| No compression | `no_compression` | — | full context, upper bound |
| FIFO | `hist_fifo` | `obs_fifo` | whole turns / head+tail truncation |
| LLMLingua | `hist_llmlingua` | `obs_llmlingua` | token-level extractive pruning, rate 0.30 |
| **Prompt-O** | `hist_prompt_o` | `obs_prompt_o` | OpenClaw two-stage checkpoint |
| **Prompt-H** | `hist_prompt_h` | `obs_prompt_h` | Hermes structured checkpoint |
| TRACE | `hist_trace` | `obs_trace` | verifier-optimised template |
| ACON-UT | `hist_acon_ut` | `obs_acon_ut` | ACON utility-optimised guideline |
| ACON-UT+CO | `hist_acon_utco` | `obs_acon_utco` | ACON utility+compression guideline |

`no_compression` is not split: it compresses nothing, so the axis does not apply,
and both axes are compared against that one row.

Rows that cannot run are refused rather than approximated. `hist_acon_utco` and
`obs_acon_utco` need the second optimiser stage, which has never been run locally,
so `check_methods.py` fails on them and `run_suite.py` skips them with a reason.

Where an upstream project ships no observation prompt — OpenClaw, Hermes and TRACE
are all history-only — the `obs.jinja` template is derived from that project's own
history prompt so its preservation criteria carry over, and the row's `source` says
so. Reusing one generic observation prompt across those methods would make their
rows identical while appearing to be different methods. The one authentic
observation prompt is ACON's own `P_obs`, used verbatim by `obs_acon_ut`.

Provenance, upstream commits and licence notes are in `docs/METHODS.md` and
`docs/CREDITS.md`.

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
python scripts/check_methods.py                      # refuse rows that cannot run
python scripts/run_suite.py --stamp my_run --shards 16
python scripts/compute_table.py --stamp my_run       # the table above
```

Needs `APPWORLD_ROOT` (or `--appworld-root`) set before `appworld` is imported, and
the endpoint reachable. `run_suite.py` resolves the served model id from
`/v1/models`, warns when the manifest disagrees, and records a behavioural
fingerprint of the backbone in every run directory so a redeployment mid-run leaves
evidence instead of silently averaging two models together.

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

The fifteen baseline rows are implemented and pass `check_methods.py` except the
two ACON-UT+CO rows, which wait on a second optimiser stage that has never been run.

Not yet exercised end to end: the engine has been written and dry-run, but no task
has been executed through it. A single-task smoke test is the gate before any full
run, because it is the only thing that exercises the agent loop, the policy call,
the artefact layout and the evaluator together.

The project's own contribution (a rate-distortion, rollout-free objective for the
compressor) is designed in `docs/DESIGN.md` and **not yet implemented** —
`ctxcomp/methods/` is where it lands, behind the same `CompressionPolicy` interface
the baselines use.
