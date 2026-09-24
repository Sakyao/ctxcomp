# Where this project's own method goes

This repository is the baseline harness. The project's own contribution is
**not implemented here yet**; its design lives in the project documents outside
this repo. This file only records the interface it has to satisfy, so that the
harness and the method can be developed without either assuming the other's
internals.

## The slot

`ctxcomp/methods/` holds one class per policy. The baselines are:

- `builtin.py` — no LLM: full history, recency truncation, token pruning.
- `prompted.py` — one code path for every prompt-driven method, because they
  differ only in prompt text; five prompt directories, one implementation.

A new method is a new class satisfying `CompressionPolicy`:

```python
def compress(self, req: CompressionRequest) -> CompressionResult: ...
```

`CompressionRequest` carries the task instruction, the compressible history, the
previous summary, the budget and `is_first`. That is the whole world the policy
sees: no environment, no tools, no evaluator, no agent transcript. Two rows are
comparable exactly to the extent that this stays true, so a method that wants to
observe more than this is a different experiment.

## What the baselines cannot express

`CompressionResult` currently returns text. Every baseline produces its
replacement by generating or discarding text. A method whose compressor is
optimised against a differentiable objective returns the same text through the
same interface; what differs is how it was produced, and that lives outside the
policy — in training, not in the harness.

That separation is the reason this repo exists independently of the training
code:

- the harness fixes the agent, the tools, the budget and the metrics, so a
  training-side change cannot move the measurements;
- the metrics are recomputed from stored trajectories, so a policy can be
  re-scored after the fact without re-running the agent.

## What is deliberately missing

- No prompt optimiser. UT/CO are inputs to this repo (`prompts/acon_ut/`), not
  part of it; running them is a separate, longer experiment and belongs with the
  training code.
- No training loop, no gradients, no model checkpointing.
- No benchmark expansion beyond AppWorld. `docs/METHODS.md` records what would
  have to change to add one, and the answer is mostly "the engine", because every
  metric here is defined on AppWorld's artefacts.
