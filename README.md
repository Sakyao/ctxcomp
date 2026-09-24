# ctxcomp — a comparison suite for context compression on AppWorld

This repository owns the **suite**: the manifest, the prompt assets, the driver, and
the results table. It does not own the **harness**. Every row is executed, unchanged,
by the code in the ACON checkout named by `ACON_ROOT`, which is where each baseline
has lived all along:

| layer | where it actually runs |
|---|---|
| compressor | `ACON_ROOT/src/productive_agents/ctxopt/{history,obs}_optimizer.py` |
| trigger | `ACON_ROOT/src/productive_agents/agents/memory.py` |
| agent loop | `ACON_ROOT/src/productive_agents/agents/unified_agent.py` |
| entry point | `ACON_ROOT/experiments/appworld/run_all.py` |

That split is deliberate. A baseline re-implemented here could differ from itself
between runs, and a comparison table cannot contain that. The only file in this
repository that adapts anything is `make_configs.py`, and only where upstream's own
generator cannot express an observation row at all — see its docstring.

## Quick start

```bash
export ACON_ROOT=/z5s/morph/home/sjk/Agent/datasets/repos/acon   # the harness

python make_configs.py                # methods.yml -> configs/<row>/<row>.yaml
bash   run_suite.sh --methods hist_fifo --shards 8      # run rows, evaluate them
python compute_table.py --stamp <stamp>                 # assemble the table
```

`run_suite.sh --dry-run` prints the plan without touching the endpoint.

## Two axes, not one

ACON compresses two different things and reports them as two separate AppWorld
lines (section 4.2 has one subsection per axis):

```
history      h'_t = f(h_t; P_hist)          if |h_t| > T_hist      T_hist = 4096
observation  o'_t = f(o_t, h_{t-1}; P_obs)  if |o_t| > T_obs       T_obs  = 256
```

They differ in input, in output shape, in threshold, and in the prompt being
optimised: `P_hist` and `P_obs` are learned separately. The axis is a property of
the **row** (`kind: history | obs`), not a suffix on a method's name.

`T_obs = 256`, not 1024. An earlier draft of this repository clamped observations at
1024; the code this suite drives clamps at 256, and a row has to be measured at the
setting it will be compared under.

What the observation axis is *not*: it is not "drop old observations". By equation
(4) it compresses the **current** observation when that observation alone is too
long, conditioned on the history so far.

## Rows

Fourteen entries in `methods.yml`; twelve produce a config, plus the uncompressed
reference which needs none.

| method | history | observation | mechanism |
|---|---|---|---|
| No compression | `no_compression` | — | full context, upper bound |
| FIFO | `hist_fifo` | — | whole turns dropped oldest-first, last 5 kept |
| LLMLingua | `hist_llmlingua` | `obs_llmlingua` | token-level extractive pruning |
| Prompting-O | `hist_prompting_o` | `obs_prompting_o` | OpenClaw two-stage checkpoint |
| Prompting-H | `hist_prompting_h` | `obs_prompting_h` | Hermes structured checkpoint |
| TRACE | `hist_trace` | `obs_trace` | verifier-optimised template |
| ACON-UT | `hist_acon_ut` | `obs_acon_ut` | ACON utility-optimised guideline |
| ACON-UT+CO | `hist_acon_utco` | `obs_acon_utco`* | ACON utility+compression guideline |

`no_compression` is not split: it compresses nothing, so the axis does not apply.

\* `obs_acon_utco` has no observation template — the local optimiser only ever
produced a history guideline — so `make_configs.py` refuses the row rather than
substituting a default.

`obs_fifo` is deliberately absent. ACON's observation axis truncates a *single*
over-long observation, and upstream implements no non-LLM way to do that. The row
could only exist by implementing a method here, which is the one thing this suite
must not do.

## Prompts: ported vs derived

`prompts/` mirrors each upstream project.

* **Ported** (byte-identical to source): all OpenClaw, Hermes and TRACE history
  prompts, ACON's `history.jinja`, and ACON's `P_obs` body
  (`prompts/acon_ut/obs.jinja`). `hist_acon_utco` is byte-identical to this
  machine's `co1/length_optimized_history_prompt_0.jinja`.
* **Derived**: `openclaw/obs.jinja`, `hermes/obs.jinja`, `trace/obs.jinja`. No
  upstream project ships an observation-level prompt, so these keep that project's
  own preservation criteria and move them onto the single-observation task. Each
  row's `source:` field says `DERIVED` and why. A derived template is a different
  object of study from the published method it is attached to, and a reader has to
  be able to tell them apart.

Two upstream adaptations are documented rather than hidden:

1. **Two-stage composition.** OpenClaw and Hermes each use a different prompt for
   the first compaction and for later ones, but the compressor exposes a single
   template plus a `prev_summary` variable that is empty on the first call. The two
   upstream prompts are composed into one template branching on `prev_summary`
   (`make_configs.py::compose_two_stage`). This is upstream's own device, recorded
   here because it is an adaptation rather than a port.
2. **The observation template key.** The history optimiser reads
   `prompt_history_user`; the observation optimiser reads `prompt_user`
   (`obs_optimizer.py:54`). Upstream's generator hard-codes the history key, so its
   `kind: obs` path cannot produce a working config. Handling both keys is the
   entire local adaptation in `make_configs.py`.

## Metrics

| column | definition | source |
|---|---|---|
| Avg Acc | mean success over repeats | AppWorld's evaluator, `evaluations/<split>.json` |
| Pass^2 / Pass@2 | solved in all / at least one run | same, requires `repeats >= 2` |
| Easy/Medium/Hard | success within each difficulty band | same |
| Steps | environment interactions per task | `analysis_tools.analyze_experiment_tokens_v2` |
| Peak | max input length over steps, **including** system prompt (10³) | same |
| Dep. | `Σ (n_i + 2·n_o)·n_o / 2`, `n_i` **excluding** system prompt (10⁶) | same |

Steps/Peak/Dep are read from `analysis_tools.analyze_experiment_tokens_v2` — the
implementation they were measured with — and are not recomputed here. Acc comes from
the evaluator, never from a log line.

## Changes made outside this repository

`patches/` records every edit made inside the ACON checkout, with the reason and the
command to revert. They are marked `[LOCAL PATCH]` in the source, so
`grep -rn "LOCAL PATCH" $ACON_ROOT` finds all of them. None of them changes a
compression method, the agent, the budget or the metric.

## Adding a method

1. Drop its prompt files under `prompts/<method>/`, recording their provenance.
2. Add a row to `methods.yml` — `name`, `label`, `kind`, `ctxopt`, and a `source:`
   that says where the prompt came from and whether it is ported or derived.
3. `python make_configs.py` — it refuses to emit a row whose prompt it cannot
   resolve, because a missing prompt does not fail loudly: the compressor falls back
   to its stock guideline and the row silently becomes a different method.

## Evaluation

`run_suite.sh` runs a row and then hands it to AppWorld's own evaluator. AppWorld
persists each task's database on every `execute`, and the evaluator replays those
databases against ground truth, so Acc is AppWorld's number and not one derived
here. Two details are easy to get wrong:

* **The evaluator walks every task the dataset lists.** A row that covered fewer
  tasks than the dataset names fails on the first uncovered one, reporting that
  task's missing `dbs` — which reads like the row failed when it was never asked to
  cover that task. `--task-ids` therefore evaluates against a generated subset
  dataset (`ctxc_<stamp>`) rather than the full split, and
  `compute_table.py --eval-dataset` reads that file. A full round uses
  `test_normal`, which every row covers.
* **The run directory is `<model>_<tag>`, not the tag.** AppWorld prefixes the
  experiment name with the model, and the evaluator is addressed by that directory
  name. Pointing it at the bare tag is a silent mismatch: the run succeeds, the
  evaluation cannot find the database, and the table shows a dash for a row that
  really did execute. `run_suite.sh` keeps tag and directory separate for this
  reason.

Verified end to end on one task (`--task-ids`), which produced
`evaluations/ctxc_<stamp>.json` and a table row carrying Acc, Steps, Peak, Dep and
the difficulty band; the full 168-task split is the same path with `test_normal`.

