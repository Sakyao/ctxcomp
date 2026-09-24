# Provenance and credits

Every prompt in `ctxcomp/prompts/` is reproduced from an upstream source, not
written here. Where a source fixes an exact commit, that commit is what was read.

| method | upstream | commit / version | licence |
|---|---|---|---|
| `prompting_o` | OpenClaw `agent-core` compaction module | `0e7b5c3` | not stated upstream |
| `prompting_h` | NousResearch `hermes-agent`, `agent/context_compressor.py` | `cca3b77` | not stated upstream |
| `trace` | TRACE, arXiv 2608.06503; template rebuilt from the released policy | `candidate_1`, `policy_sha256 94d193fc…` | **none — see below** |
| `acon_ut` | our own UT run over the `microsoft/acon` optimiser | `d63f9ae` | MIT |
| `llmlingua` | `microsoft/LLMLingua` | `v0.2.2` (`a411a3fa`) | MIT |
| `fifo` | baseline defined in ACON (Kang et al., ICML 2026) | — | — |

## TRACE

The TRACE repository (`nokia-applied-research/Trace`) carries **no licence file**,
and its own README states this and advises adding one before distributing. The
template used here was therefore rebuilt from the released policy data with the
authors' own renderer, and is attributed to the paper rather than copied from the
repository. Do not vendor that repository's code into this one.

## AppWorld

The environment and its evaluator belong to AppWorld
(`StonyBrookNLP/appworld`), Apache-2.0. Trajectory data produced here is a derived
work of it.

## What is not reproduced

The baselines that the two source papers train — AgentFold, Context-Folding, MEM1,
MemAgent, Agent-Omit — change the backbone or its policy, so they are not rows
here. Retrieval is deliberately absent: selecting earlier turns is memory
selection rather than long-horizon compression, and TRACE, the closest work in
this setting, omits it as a baseline too.
