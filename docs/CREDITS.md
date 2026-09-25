# Provenance and credits

Every prompt in `prompts/` comes from an upstream source or is marked as derived
here. Where a source fixes an exact commit, that commit is what was read.

| method | axis | upstream | commit / version | licence | template |
|---|---|---|---|---|---|
| `prompting_o` | history | OpenClaw `agent-core` compaction module | `0e7b5c3` | not stated upstream | ported |
| `prompting_h` | history | NousResearch `hermes-agent`, `agent/context_compressor.py` | `cca3b77` | not stated upstream | ported |
| `trace` | history | TRACE, arXiv 2608.06503; template rebuilt from the released policy | `candidate_1`, `policy_sha256 94d193fc…` | **none — see below** | ported |
| `acon_ut` | history | local UT run over the `microsoft/acon` optimiser | `d63f9ae`, output `ut1` | MIT | ported |
| `acon_utco` | history | local CO run, same optimiser | output `co1` | MIT | ported |
| `llmlingua` | history, obs | `microsoft/LLMLingua` | `v0.2.2` (`a411a3fa`) | MIT | — |
| `fifo` | history | baseline defined in ACON (Kang et al., ICML 2026) | — | — | — |
| `acon_ut` | obs | ACON's `P_obs`, `experiments/appworld/prompts/context_opt/prompt_user.jinja` | `d63f9ae` | MIT | ported |
| `prompting_o` | obs | — | — | — | **derived** |
| `prompting_h` | obs | — | — | — | **derived** |
| `trace` | obs | — | — | — | **derived** |

The three derived observation templates keep their source project's preservation
criteria and move them onto the single-observation task, because no upstream project
ships an observation-level prompt. They are different objects of study from the
published methods they are attached to; `methods.yml` says `DERIVED` in each row's
`source:`.

## TRACE

The template used here is `candidate_1` from the TRACE repository
(`nokia-applied-research/Trace`) — the one the paper's authors took to their
end-to-end runs — rendered with their own renderer and checked against it:

| our file | produced by | check |
|---|---|---|
| `prompts/trace/update.jinja` | `trace_cc.optimize.policy.build_update_template(candidate_1.json, base_update_text())` | byte-identical |
| `prompts/trace/first.jinja` | `data/compression_policy_base/first_summary.jinja` | byte-identical |
| `prompts/trace/system_prompt.jinja` | `data/compression_policy_base/system_prompt.jinja` | byte-identical |

`policy_sha(candidate_1)` is `94d193fcc6f96e12027b18212bce88bc320dcb55f31d34bd2ec4bd976d6c59ea`,
the hash that repository's `manifest.json` records. The two base templates are also
byte-identical to the ones under `prompts/openclaw/`, which is what the paper means by
its harness "adapting OpenClaw's recurrent compaction loop": `prompting_o` and `trace`
are the same loop with and without the eleven optimised slots.

The repository carries **no licence file**, and its own README says so and advises
adding one before distributing. Nothing from it is vendored here: the template was
rendered from the published policy and is attributed to the paper.

### Execution-layer differences from the TRACE paper

This row runs on the suite's shared harness, not on TRACE's own. After reading their
reference implementation (`trace_cc/core.py`, `optimize/loop.py`,
`collect/runner.py`), the divergences are:

| | TRACE | here |
|---|---|---|
| compression threshold | 4096 | 4096 |
| turns kept raw | `preserve_last_k_turns = 1` | 1 |
| first vs update | `prev_summary is None` → `first_summary.jinja`, and "no previous summary is ever fabricated" | `prev_summary == ''` → `{% if prev_summary %}` branch |
| summary wrapper | `<history_summary>…</history_summary>` | `<HISTORY_SUMMARY>…</HISTORY_SUMMARY>` |
| history rendering | `reasoning` + fenced code + `Output:` | `USER:` / `ASSISTANT:` |

The last two rows are the real divergence, and they are deliberately not corrected: a
row executed by its own private loop would no longer be comparable with the other rows
in the table. TRACE's own `collect/runner.py` states the constraint this reproduces —
rollouts through "a different adapter will not be byte-identical", because "prompt
assembly and tool-call formatting differ". TRACE numbers from this suite and from the
paper are therefore two cohorts, not two measurements of one thing.

## AppWorld

The environment and its evaluator belong to AppWorld (`StonyBrookNLP/appworld`),
Apache-2.0. Trajectory data produced here is a derived work of it.

## ACON harness

The executables this suite drives -- `productive_agents.ctxopt`, `agents.memory`,
`agents.unified_agent`, `experiments/appworld/run_all.py` -- belong to
`microsoft/acon` (MIT), at `d63f9ae` with the local edits listed in `patches/`.

## Where the ACON guidelines come from

The four ACON rows carry the **published** guidelines, not a local optimiser run. The
paper's Appendix E prints the guideline for each axis at each stage, and those are the
objects the method is defined by:

| row | paper prompt | file |
|---|---|---|
| `hist_acon_ut` | E.6 — history after optimization (UT) | `prompts/acon_ut/history.jinja` |
| `hist_acon_utco` | E.7 — history after optimization (UTCO) | `prompts/acon_utco/history.jinja` |
| `obs_acon_ut` | E.9 — observation after optimization (UT) | `prompts/acon_ut/obs.jinja` |
| `obs_acon_utco` | E.10 — observation after optimization (UTCO) | `prompts/acon_utco/obs.jinja` |
| *(reference)* | E.5 — history before optimization | `prompts/acon_base/history.jinja` |
| *(reference)* | E.8 — observation before optimization | `prompts/acon_base/obs.jinja` |

Reproduced verbatim: PDF line breaks rejoined, page numbers and headers removed,
nothing paraphrased. Appendix E.1–E.4 are the optimiser's own analysis and update
prompts — machinery, not guidelines — and are not used here.

The template files carry **no annotation of their own** — no provenance header, no
comment block. They are the paper's text and nothing else, so a byte-level comparison
against Appendix E is meaningful. Their provenance (which Prompt, which axis, which
stage) is recorded here and in each row's `source:` in `methods.yml`, never inside the
templates.

**Superseded local artefacts.** An earlier state of this repository used candidates
from a local optimiser run. Those candidates had never been through the stage the
paper describes ("sample 5 candidate prompts and select the one that performs best on
a subset of the training set"), so they were not the method. They are archived rather
than deleted, so the substitution is auditable:

| file | sha256 |
|---|---|
| `prompts/_superseded/ut1_sample0_history.jinja` | `62f6b1769c4cd1cb9fc9694a0fd038c3e2acea414f97167650ff5f51e03ae7d6` |
| `prompts/_superseded/co1_sample0_history.jinja` | `ff5239127bf8aadedad55769590c591766c0df08c9c576c27992b37973d3bf81` |

### A divergence between the paper and the code it ships with

Paper Appendix B.3 sets `T_hist = 4096` and `T_obs = 1024` for AppWorld. Upstream's
own config generator hard-codes **256** on the observation axis — four times more
aggressive. These rows use the paper's values (see the note at the top of
`methods.yml`), because the table is meant to be read against the paper. Anything
measured at 256 is a different measurement and must not be averaged into this one.

## What is not reproduced

The baselines that the two source papers train -- AgentFold, Context-Folding, MEM1,
MemAgent, Agent-Omit -- change the backbone or its policy, so they are not rows
here. Retrieval is deliberately absent: selecting earlier turns is memory selection
rather than long-horizon compression, and TRACE, the closest work in this setting,
omits it as a baseline too.
