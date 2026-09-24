# Design: why this repository is a suite, not a harness

## The split

```
ctxcomp (this repository)                 ACON checkout (ACON_ROOT)
  methods.yml      the manifest             src/productive_agents/ctxopt/   compressors
  make_configs.py  manifest -> co_config    src/productive_agents/agents/memory.py    trigger
  run_suite.sh     shards, fingerprint,     src/productive_agents/agents/unified_agent.py  loop
                   run, evaluate            experiments/appworld/run_all.py entry point
  compute_table.py the table                experiments/analysis_tools/    metric definitions
  prompts/         prompt assets
```

The suite knows what to compare and how to read the result; the harness knows how to
run a policy. Every row is executed by the baseline's own code, so a baseline cannot
drift from its published implementation by being re-expressed here.

This was not the first shape of the repository. An earlier draft re-implemented the
agent loop, the compressor dispatch, the metric reducer and the LLMLingua client,
on the reasoning that a single uniform engine makes the rows comparable. It does the
opposite: it makes every row a measurement of the re-implementation, and the places
where the copy and the original disagree — the observation budget, the client
protocol, whether the system prompt counts toward Peak — are invisible in the
output. The rule this repository now follows is that a difference between two rows
may come from the method or from the prompt, and from nothing else.

## What the suite still owns, and why that is not a re-implementation

- **The manifest.** Which rows exist, on which axis, with which prompt, and where
  each prompt came from. No harness can answer that; it is the experiment's design.
- **Config generation.** One YAML per row, with the prompt directory copied beside
  it so a config is self-contained and diffable. The only adaptation here is that
  the observation axis needs the `prompt_user` key while upstream's generator emits
  `prompt_history_user`, which makes its own `kind: obs` path non-functional.
- **The driver.** Sharding, the endpoint fingerprint taken before and after each row,
  run naming, and invoking AppWorld's evaluator. A row whose fingerprint moved is
  stamped `CONTAMINATED` rather than silently averaged in.
- **The table.** Reads Acc from AppWorld's evaluator output and Steps/Peak/Dep from
  `analysis_tools.analyze_experiment_tokens_v2`, the implementation they were
  measured with. It computes the pass^k aggregates and the difficulty split, and
  prints `-` where a number does not exist.

## What is deliberately missing

- **No prompt optimiser.** UT/CO are inputs (`prompts/acon_ut/`,
  `prompts/acon_utco/`), not part of the suite; running them is a separate, longer
  experiment.
- **No training loop,** no gradients, no checkpointing.
- **No engine of our own.** If a baseline cannot be run by the ACON harness, the row
  is not run and `methods.yml` says why. `obs_fifo` is the worked example: ACON has
  no non-LLM observation truncation, so implementing one here would produce a row
  labelled like a baseline and measured like an invention.

## Metrics, and where they come from

| column | source |
|---|---|
| Acc, Pass^2, Pass@2, Easy/Medium/Hard | AppWorld evaluator, `evaluations/<split>.json` |
| Steps, Peak, Dep. | `analysis_tools.analyze_experiment_tokens_v2` |

Peak includes the system prompt; Dep. excludes it. Both are read from the stored
trajectory rather than from a counter written during the run, so a row can be
re-scored after the fact and the definition is auditable in one place.

## Adding a benchmark

Every metric here is defined on AppWorld's artefacts — its evaluator, its difficulty
labels, its interaction count. A second benchmark is not a flag; it is a second
`ACON_ROOT`-side entry point plus a metric reader, and the honest version of that
work starts by finding out whether the harness already exposes those three things.
