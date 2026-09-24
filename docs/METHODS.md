# Methods

Every row is a compression policy. The agent, its prompt, its tools, its decoding
and the environment are identical across rows; only the representation handed
back to the agent differs.

| method | kind | mechanism | prompt |
|---|---|---|---|
| `no_compression` | none | full history | — |
| `fifo` | history | discard whole turns, oldest first, keep the last k | — |
| `llmlingua` | history | token-level extractive pruning, keep rate 0.30 | — |
| `prompting_o` | history | two-stage structured checkpoint (first / update) | `prompts/openclaw/` |
| `prompting_h` | history | structured checkpoint + reference-only prefix | `prompts/hermes/` |
| `trace` | history | verifier-optimised checkpoint template | `prompts/trace/` |
| `acon_ut` | history | ACON utility-optimised guideline | `prompts/acon_ut/` |
| `acon_utco` | history | ACON utility+compression-optimised guideline | `prompts/acon_utco/` (absent) |

## UT and CO

ACON optimises the compressor's natural-language guideline in two alternating
stages:

- **UT — utility maximisation.** Sample candidate prompts, keep the one with the
  best success rate. This teaches the compressor not to drop information the task
  needs. Output files are prefixed `improved_history_prompt_`.
- **CO — compression maximisation.** Run only on the tasks that already succeed
  under the current guideline, and let the model report which information the
  execution actually used, so the checkpoint can be shortened without giving back
  success rate. Output files are prefixed `length_optimized_history_prompt_`.

`acon_ut` here is our own UT run on the AppWorld train split. **It is not the
guideline the ACON authors shipped**, and it cannot be: UT/CO output depends on
the trajectories and the agent prompt it was optimised against, so two faithful
reproductions of the method produce two different texts and therefore two
different rows. Any result should be reported with the guideline's sha256 so that
two papers using the same method name are not mistaken for the same row.

`acon_utco` needs the second stage's output. Only the UT stage has been run
locally, so the directory is empty and both `check_methods.py` and
`run_suite.py` refuse the row rather than quietly substituting a default.

## Two-stage compaction

OpenClaw and Hermes each use a different prompt for the first compaction and for
later ones. `PromptedPolicy` selects between `first.jinja` and `update.jinja` on
`is_first`, which is native here rather than composed, so the two upstream prompts
keep their original shape.

## Notes that change how numbers should be read

- **LLMLingua is applied to the history**, not to observations, so the row is
  comparable with the ACON baseline rather than with observation-level pruning.
- **FIFO keeps whole turns.** It does not keep the action and drop the
  observation — that is observation masking, a different method, and conflating
  them would make both rows uninterpretable.
- **`api_docs_mode`** decides whether the per-task API documentation is pasted in
  whole. It is ~100k tokens for a sampled task, and it is the reason the
  uncompressed row is expensive. The setting is recorded in `results.json`.

## The two axes

ACON compresses the history and the observations, and reports them as two separate
AppWorld lines. They are not two settings of one method:

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
  just read. A table that mixes them along one axis is measuring two different
  interventions.
- **Observation compression is not observation masking or a recency window.** By
  equation (4) it acts on the current observation and conditions on the history, so
  the mechanism has to reduce one observation's length. Dropping old observations
  is a different method, and implementing it here would mislabel the row.

### Where the observation prompts come from

Only ACON defines an observation prompt. `obs_acon_ut` uses it verbatim
(`prompts/acon_ut/obs.jinja`, reproduced from
`experiments/appworld/prompts/context_opt/prompt_user.jinja`).

OpenClaw, Hermes and TRACE are all history-only, so `obs_prompting_o`, `obs_prompting_h`
and `obs_trace` use templates derived from each project's own history prompt,
keeping its preservation criteria and moving them onto the single-observation task.
Each derived file carries a header saying so, and so does the row's `source`. The
alternative — one generic observation prompt shared by three methods — would make
three rows identical while presenting them as three different methods, which is
worse than an honest derivation.
