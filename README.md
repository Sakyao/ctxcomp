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
| **Prompting-O** | `hist_prompting_o` | `obs_prompting_o` | OpenClaw two-stage checkpoint |
| **Prompting-H** | `hist_prompting_h` | `obs_prompting_h` | Hermes structured checkpoint |
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

## Where results go

One method-run produces one folder. Everything a run produces is in it:

```
runs/<timestamp>/<method>__<served_model>/
    run.json                          method, axis, served model, backbone
                                      fingerprint, verdict, seconds, Acc
    evaluations/<split>.json          AppWorld's scorer output
    evaluations/<split>.txt           the same, human-readable
    tasks/<task_id>/
        env_history.json              one entry per environment interaction
        llm_history.json              per-call messages; Peak and Dep read this
        token_usage_and_cost.json
        results.json                  steps, cap hit, compaction counts, config echo
        dbs/  logs/  checkpoints/     AppWorld's own: the final database state the
        misc/ version/                evaluator replays to decide success
```

The metric files sit beside `dbs/` rather than in a tree of their own, because they
are the same run. ACON's harness scattered them across two roots and bridged them
with a symlink, which is why Acc and Steps/Peak/Dep could not be read from one
place; that is not repeated here.

`<timestamp>` defaults to `YYYYMMDD_HHMMSS` so successive rounds never overwrite
each other. `<served_model>` is the id the endpoint actually serves, not what the
manifest asked for — the server rejects names it has not loaded.

AppWorld's evaluator will only look under `<APPWORLD_ROOT>/experiments/outputs`, and
`experiment_name` may be a relative path, so runs are named `<stamp>/<run_name>` and
a symlink at `<APPWORLD_ROOT>/experiments/outputs/<stamp>` resolves back into this
repository. The results are stored once; the second path is a link.

The behavioural fingerprint of the backbone is recorded in `run.json`, not in the
directory name: it is evidence of *which weights answered* — the only value that
distinguished two redeployments sharing a model name — but it changes whenever the
serving configuration does, and a directory that renamed itself mid-experiment
would be unusable.

## Metrics

| metric | definition |
|---|---|
| **Avg Acc** | mean task success over `repeats` independent runs |
| **Pass^2** | fraction solved in **all** runs — multi-run reliability |
| **Pass@2** | fraction solved in **at least one** run — task coverage |
| **Easy / Medium / Hard** | per-difficulty success (57 / 48 / 63 tasks) |
| **Steps** | mean environment interactions per task |
| **Peak** | max input context length over all steps, **including** the system prompt, in 10³ tokens |
| **Dep.** | `Σ_t ((n_i + 2·n_o)·n_o)/2` with `n_i` **excluding** the system prompt, in 10⁶ |

Acc, difficulty and the evaluator's verdict come from AppWorld's scorer, which
replays the final database state against ground truth. Steps, Peak and Dep are
recomputed here from the stored trajectories, so a past run can be re-scored without
re-running the agent and a change in definition does not silently mix with old
numbers.

## Layout

```
methods.yml              the single source of truth: one entry per method
ctxcomp/
  methods/               one CompressionPolicy per (method, axis)
  prompts/               prompt templates, one directory per prompt-driven method
  metrics/tokens.py      Steps / Peak / Dep
  metrics/table.py       the table above
  metrics/difficulty.py  the environment's own Easy/Medium/Hard labels
  runner/                AppWorld execution loop, both axes
scripts/                 check_methods.py, run_suite.py, compute_table.py
docs/                    methods, provenance, credits, design
runs/                    results, one folder per round (gitignored)
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
