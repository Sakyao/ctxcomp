# Vendored harness — where it came from and how to re-sync it

This repository runs the comparison experiments itself. It does not import a second
checkout at run time. The code that executes each baseline is **copied here**,
byte-identical to upstream, so a round can be reproduced from this repository alone
and the upstream checkout stays what it should be: the reference the copy is diffed
against.

## Source

| | |
|---|---|
| upstream | `/z5s/morph/home/sjk/Agent/datasets/repos/acon` (`microsoft/acon`) |
| commit | `d63f9ae18959dc7215ff62899c94c5e8c56847ae` |
| licence | MIT — copy at `vendor/LICENSE.acon` |
| vendored at | 2026-09-25 |

## What was copied

| destination | source | files | what it is |
|---|---|---|---|
| `src/productive_agents/` | `src/productive_agents/` | 86 `.py` | compressors (`ctxopt/`), memory manager (`agents/memory.py`), agent loop (`agents/unified_agent.py`), LLM clients (`llm.py`) |
| `experiments/appworld/` | `experiments/appworld/` | `run_all.py`, `run.py`, `run_ctxopt_pipeline.py`, `configs/`, `prompts/`, `scripts/`, `README.md` | the entry point, the agent prompt, the local compressor configs |
| `experiments/analysis_tools/` | `experiments/analysis_tools/` | `utils.py` | `analyze_experiment_tokens_v2` — the Steps/Peak/Dep implementation |
| `experiments/repro/` | `experiments/repro/` | `llmlingua_server.py`, `fingerprint_endpoint.py`, `embedding_server.py`, `ENDPOINT_IDENTITY.md` | the LLMLingua HTTP service and the backbone fingerprint probe |
| `experiments/__init__.py` | same | 1 | marks `experiments` a package for `from experiments.analysis_tools...` |

120 files total, excluding `__pycache__` and `*.pyc`. The copy is verbatim: `diff -rq`
against the source reports nothing for any of these trees.

**Not copied** — build artefacts and data, none of which are code: `outputs/` (528 MB
of past runs), `experiments/` (95 MB of evaluator reports), `shards/`, `data/` (a
symlink to the AppWorld data root), `data_copy/`. Run artefacts are regenerated per
round and are git-ignored.

## Local modifications travel with it

The upstream checkout carries five `[LOCAL PATCH]` edits (see `patches/README.md`:
pricing fallback, configurable endpoint, retry, local embedding client, retrieval
cache). They are load-bearing — without them every run against a self-hosted model
raises `KeyError` before the first step — so the vendored copy includes them as they
stand in the working tree, **not** as pristine upstream. `grep -rn "LOCAL PATCH"
src experiments` lists every one of them here too.

## Fingerprint

Stable digest of the vendored trees, so drift is detectable:

```bash
cd <repo root>
find src experiments -type f -not -path "*__pycache__*" -not -name "*.pyc" \
  | sort | xargs sha256sum | sha256sum
```

As vendored on 2026-09-25:

```
2c6026ec2419852b69eec901fb77e5761922e2393502220f464c8c388a19ad0d
```

If a run's numbers are ever questioned, this digest plus the upstream commit is what
identifies the code that produced them.

## Re-syncing

```bash
bash vendor/sync_from_acon.sh          # re-copy from the upstream checkout
```

The script mirrors the trees with `rsync --delete`, so anything added under `src/` or
the mirrored directories of `experiments/` locally will be removed. That is the
intended semantics for a vendored copy: if a change here is meant to survive, it
belongs in the suite (this repository's own files) or in `patches/`, not inside the
vendored trees.

After a re-sync, re-check that the suite still runs (`bash run_suite.sh --dry-run`)
and update the digest above.
