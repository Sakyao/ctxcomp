# patches/ — changes made outside this repository

This repository is meant to hold the project's code. This directory exists because
three of its dependencies had to be modified in order to run at all, and a
modification that lives invisibly inside someone else's working tree is a
reproducibility hazard: the numbers it produces cannot be attributed either to the
method under study or to the local edit.

So the edits are recorded here as patches, with reasons, and with the commands to
restore upstream or re-apply them.

---

## 1. `acon_local_changes.patch`

**5 tracked files modified, +208 / −56**, against `microsoft/acon` at `d63f9ae`.
Every hunk is marked `[LOCAL PATCH]` in the source, so an audit of the reading
`grep -rn "LOCAL PATCH"` finds all of them.

| file | change | why it was necessary |
|---|---|---|
| `src/productive_agents/llm.py` (+60/−29) | **① pricing lookup** — upstream falls back to `MODEL_PRICING['gpt-4o']` on an unknown model name, but that key does not exist in its own table, so the fallback raises `KeyError`. Any run against a model name upstream never heard of (i.e. every self-hosted model) dies immediately. Patched to treat an unknown model as zero-cost, which is also the factually right answer for a local endpoint. | **load-bearing** |
| | **② endpoint configurable** — the vLLM branch had its base URL hard-wired. Made configurable so it can point at the local OpenAI-compatible service. | load-bearing |
| | **③ retry** — the OpenAI-compatible branch had no retry; one transient failure killed the whole task. | affects yield, not method |
| `src/productive_agents/subtrate_api.py` (+38/−2) | added `_LocalEmbeddingClient`, an OpenAI-compatible embeddings client. Upstream talks to Azure only, so with a local endpoint the **Retrieval baseline could not run at all**. | load-bearing for `*_retrieval` |
| `src/productive_agents/ctxopt/history_optimizer.py` (+91/−20) | embedding cache re-keyed from position to content, plus the candidate-pool de-duplication. A positional cache is only valid while the pool grows by appending; de-duplication shortens it. | correctness of the retrieval baseline |
| `src/productive_agents/agents/memory.py` (+11/−4) | retrieval selection by embedding similarity rather than a recency window. | makes the row mean what its name says |
| `experiments/appworld/run_all.py` (+8/−1) | expose `--prompt_file`. Upstream hard-coded `./prompts/prompts_v1.json`. The default is **byte-identical** to upstream, so omitting the flag reproduces stock behaviour exactly. | needed to select the answer-fixed prompt |

### Why these are not simply reverted

Reverting returns `acon` to pristine, and **also** removes the ability to run it:

- without the `llm.py` pricing patch, **every** run against the local endpoint
  raises `KeyError` before the first step;
- without `subtrate_api.py`, the Retrieval and observation baselines cannot start;
- without `history_optimizer.py`, the retrieval row measures a recency window.

None of these patches change the compression method, the agent, the budget or the
metric. They make the harness executable against a self-hosted model and a local
embedding service instead of Azure. That is what must be stated in the paper's
implementation notes; they are not a licence to modify baselines.

### Restore upstream, or re-apply

```bash
# back to pristine upstream
cd /z5s/morph/home/sjk/Agent/datasets/repos/acon
git checkout -- src/productive_agents/llm.py \
                src/productive_agents/subtrate_api.py \
                src/productive_agents/ctxopt/history_optimizer.py \
                src/productive_agents/agents/memory.py \
                experiments/appworld/run_all.py

# re-apply later
git apply /z5s/morph/home/sjk/Agent/datasets/repos/ctxcomp/patches/acon_local_changes.patch
```

> Caveat on the patch file: it is a snapshot of the working tree at the time it was
> taken. If `acon` is edited again, regenerate it, or the patch and the tree will
> disagree.

---

## 2. `acon_untracked_items.txt`

Ten new (untracked) items were added under `acon`, i.e. files upstream does not have.
They are listed in full in that file. Broadly:

| item | what it is | keep? |
|---|---|---|
| `experiments/appworld/prompts/prompts_v1_answerfix.{json,jinja}` | the answer-fixed agent prompt | **load-bearing** — every run that used it needs it to be reproducible |
| `experiments/appworld/configs/context_opt/*.yaml` | local compressor configs (`fifo_keep5`, `history`, and the UT guideline config) | load-bearing |
| `experiments/appworld/scripts/run_acon_ut_pipeline.sh` | driver for the UT optimiser pipeline | keep — it is how `outputs/ut1/` was produced |
| `experiments/appworld/scripts/run_paper_baselines.sh` | baseline driver | keep |
| `experiments/appworld/shards/` | task-id shard files | generated, safe to delete |
| `experiments/repro/` | orchestration for the earlier comparison round | historical |
| `experiments/suite/` | **superseded prototype** of what became this repository | **should move here or be deleted** |

The intended end state is that nothing under `acon` is ours except what the
optimiser needs to be re-runnable, and that everything else lives here.

---

## 3. `appworld` and `trace`

**No modifications.** The `appworld` clone
(`Agent/datasets/repos/appworld`, `StonyBrookNB/appworld` at `42b5bcf`) reports
zero modified tracked files. Runtime artefacts go to a *different* path,
`Agent/datasets/appworld-0.1.0/`, which is not a git repository at all — it is the
environment's data and output root, addressed through `APPWORLD_ROOT`.

No `trace` repository is checked out anywhere under this home directory.

---

## 4. `experiments/suite/` has moved here — no new patch

Item 2 above listed `experiments/suite/` as a superseded prototype that "should move
here or be deleted". It has moved here: `methods.yml`, `make_configs.py`,
`run_suite.sh` and `compute_table.py` now live at this repository's root, and the
suite drives the ACON checkout through `ACON_ROOT` instead of sitting inside it.

**This adds no modification to ACON.** The patch set stays the five files in item 1;
`grep -rn "LOCAL PATCH"` still finds all of them. The only things this repository
writes into the ACON tree are runtime artefacts: the generated subset datasets under
`data/datasets/` (`ctxc_<stamp>.txt`, used to evaluate a partial run against a
partial dataset) and the run output ACON already wrote there.

Two defects in the old prototype were fixed during the move, both in this
repository's copies and neither in ACON:

- `methods.yml` named `./prompts/prompts_v1_answerfix.jinja`, which does not exist;
  the file that works is the JSON, and a real run in this checkout records
  `./prompts/prompts_v1_answerfix.json`.
- `run_suite.sh` evaluated by the bare tag while AppWorld's directory is
  `<model>_<tag>`, so evaluation could never find the run; and its
  `run_suite.sh`-generated table pattern required a `-` before the row name, which
  axis-qualified names (`hist_*`, `obs_*`) do not have.
