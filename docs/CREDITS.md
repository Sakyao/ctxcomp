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

The TRACE repository (`nokia-applied-research/Trace`) carries **no licence file**,
and its own README states this and advises adding one before distributing. The
template used here was therefore rebuilt from the released policy data with the
authors' own renderer, and is attributed to the paper rather than copied from the
repository. Do not vendor that repository's code into this one.

## AppWorld

The environment and its evaluator belong to AppWorld (`StonyBrookNLP/appworld`),
Apache-2.0. Trajectory data produced here is a derived work of it.

## ACON harness

The executables this suite drives -- `productive_agents.ctxopt`, `agents.memory`,
`agents.unified_agent`, `experiments/appworld/run_all.py` -- belong to
`microsoft/acon` (MIT), at `d63f9ae` with the local edits listed in `patches/`.

## What is not reproduced

The baselines that the two source papers train -- AgentFold, Context-Folding, MEM1,
MemAgent, Agent-Omit -- change the backbone or its policy, so they are not rows
here. Retrieval is deliberately absent: selecting earlier turns is memory selection
rather than long-horizon compression, and TRACE, the closest work in this setting,
omits it as a baseline too.
