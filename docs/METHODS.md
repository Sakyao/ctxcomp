# Methods

`methods.yml` is the authoritative list; this file explains what the rows mean. The
agent, its prompt, its tools, its decoding and the environment are identical across
rows — only the representation handed back to the agent differs, and each row is
executed by that baseline's own implementation inside the ACON checkout.

## Rows

| row | kind | mechanism | template |
|---|---|---|---|
| `no_compression` | none | full history | — |
| `hist_fifo` | history | discard whole turns, oldest first, keep the last 5 | — |
| `hist_llmlingua` | history | LLMLingua-2 extractive pruning | — |
| `hist_prompting_o` | history | OpenClaw two-stage checkpoint (first / update) | ported |
| `hist_prompting_h` | history | Hermes structured checkpoint + reference-only note | ported |
| `hist_trace` | history | TRACE verifier-optimised checkpoint template | ported |
| `hist_acon_ut` | history | ACON utility-optimised guideline (paper E.6) | ported |
| `hist_acon_utco` | history | ACON utility+compression-optimised guideline (paper E.7) | ported |
| `obs_llmlingua` | observation | same pruner, on the current observation | — |
| `obs_prompting_o` | observation | OpenClaw criteria applied to one observation | **derived** |
| `obs_prompting_h` | observation | Hermes criteria applied to one observation | **derived** |
| `obs_trace` | observation | TRACE criteria applied to one observation | **derived** |
| `obs_acon_ut` | observation | ACON's own `P_obs` | ported |
| `obs_acon_utco` | observation | ACON utility+compression-optimised `P_obs` (paper E.10) | ported |

`no_compression` has no axis: it compresses nothing, so both axes are compared
against this one row. `obs_fifo` is not run — see the end of `methods.yml`.

## UT and CO

ACON optimises the compressor's natural-language guideline in two alternating
stages:

- **UT — utility maximisation.** Sample candidate prompts, keep the one with the
  best success rate, so the compressor stops dropping information the task needs.
  Output files are prefixed `improved_history_prompt_`.
- **CO — compression maximisation.** Run only on the tasks that already succeed
  under the current guideline, and let the model report which information the
  execution actually used, so the checkpoint can be shortened without giving back
  success rate. Output files are prefixed `length_optimized_history_prompt_`.

All four ACON rows carry the **published** guidelines from the paper's Appendix E --
Prompt E.6/E.7 for history, E.9/E.10 for observation -- reproduced verbatim. The
unoptimised starting points, E.5 and E.8, are under `prompts/acon_base/`, so the
before/after difference can be read without the paper.

An earlier state of this repository used candidates from a local optimiser run
(`ut1`, `co1`). Those had never been through the stage the paper describes -- "sample
5 candidate prompts and select the one that performs best on a subset of the training
set" -- so they were not the method; `ut1`'s stage 3 was interrupted after one of five
candidates and never produced a winner. They are archived under `prompts/_superseded/`
with their sha256 in `docs/CREDITS.md` rather than deleted, so the substitution is
auditable.

## The two axes

| | history | observation |
|---|---|---|
| equation | (3) `h'_t = f(h_t; P_hist)` | (4) `o'_t = f(o_t, h_{t-1}; P_obs)` |
| fires on | the accumulated history | the **current** observation |
| fires when | `|h_t| > T_hist` | `|o_t| > T_obs` |
| threshold | 4096 | 1024 |
| prompt input | `(task, prev_summary, history)` | `(task, history, observation)` |
| prompt output | a structured checkpoint | a refined observation |

`P_hist` and `P_obs` are optimised separately, so a row is a (method, axis) pair.

Two consequences worth keeping in mind:

- **A history row and an observation row are not comparable.** The history row
  replaces what the agent remembers; the observation row replaces what the agent
  just read. A table that mixes them along one axis measures two different
  interventions.
- **Observation compression is not observation masking or a recency window.** By
  equation (4) it acts on the current observation and conditions on the history, so
  the mechanism has to reduce one observation's length.

## Two-stage compaction

OpenClaw and Hermes each use a different prompt for the first compaction and for
later ones, but the compressor exposes a single template plus a `prev_summary`
variable that is empty on the first call. `make_configs.py::compose_two_stage`
composes the two upstream prompts into one template branching on `prev_summary`. This
is the upstream harness's own device, not a local invention, but it is recorded as an
adaptation rather than a port.

## Notes that change how numbers should be read

- **LLMLingua's keep rate is not 30% in practice.** `keep_rate` appears in
  upstream's manifest but is never read: `history_optimizer.py:117` hard-codes
  `ratio=0.2` and `obs_optimizer.py:99` hard-codes `0.3`. The history row is
  therefore a 20% keep rate.
- **FIFO keeps whole turns.** It does not keep the action and drop the observation —
  that is observation masking, a different method, and conflating them would make
  both rows uninterpretable.
