"""Reference execution loop: one AppWorld task, one compression policy.

Deliberately owned here rather than delegated, because the compaction boundary is
the object of study. The loop is:

    build context -> ask the model -> execute the code -> record the turn
    -> if the compressible history exceeds the budget, hand it to the policy and
       continue from the policy's replacement text

The policy never sees the environment, the tools or the evaluator; it only sees
the history and returns text. Everything else is frozen, which is what makes two
rows comparable.

Two notes on fidelity, because they are the difference between this loop and the
harness the baselines were originally measured on:

* `api_docs` is per task and can be very large (425 KB for one sampled task,
  roughly 100k tokens). `api_docs_mode: full` reproduces a harness that pastes it
  in whole, which is what makes the uncompressed row genuinely expensive;
  `instruction-only` is the cheap variant. The choice is recorded in results.json
  because it changes the meaning of Peak and Dep.
* Compaction is triggered on characters here, not tokens, to keep the boundary
  dependency-free. The budget is therefore approximate; it is recorded with every
  run so that a later exact-token implementation can be told apart from this one.

Artifacts are written in the shape ctxcomp.metrics expects:
    env_history.json              one entry per environment interaction
    llm_history.json              per-call message lists (Peak and Dep read this)
    token_usage_and_cost.json     call and token counters
    results.json                  steps, cap, compaction count, config echo
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..llm import ChatClient
from ..methods.base import CompressionPolicy, CompressionRequest, Turn
from .base import EngineConfig, TaskOutcome

CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.S)

DEFAULT_AGENT_PROMPT = """You are an autonomous agent solving a task by writing Python code.

At each step you are given the task instruction, the available API documentation,
and the history of what you have already done. Reply with a single Python code
block and nothing else. The code runs in a persistent interpreter, so variables
you define stay available.

Rules:
- Call the APIs exactly as documented. Parameter names matter.
- `apis.supervisor.complete_task(...)` ends the task. For tasks that are not
  questions, call it with no `answer` argument; passing an answer to a
  non-question task fails the task.
- Do not re-do work that already succeeded.
"""


def extract_code(text: str) -> str:
    m = CODE_BLOCK.search(text or "")
    return (m.group(1) if m else (text or "")).strip()


def build_context(instruction: str, api_docs: str, history: list[Turn],
                  summary: str, cfg: EngineConfig) -> str:
    parts = [f"# Task\n{instruction}"]
    if api_docs:
        parts.append(f"# API documentation\n{api_docs}")
    if summary:
        parts.append(f"# Summary of earlier work\n{summary}")
    if history:
        recent = "\n\n".join(f"ACTION:\n{t.action}\n\nOBSERVATION:\n{t.observation}"
                             for t in history)
        parts.append(f"# Recent steps\n{recent}")
    parts.append("# Next step\nReply with one Python code block.")
    return "\n\n".join(parts)


class AppWorldEngine:
    def __init__(self, llm: ChatClient | None = None):
        self.llm = llm or ChatClient()

    def run_task(self, task_id: str, policy: CompressionPolicy | None,
                 cfg: EngineConfig) -> TaskOutcome:
        from appworld import AppWorld

        world = AppWorld(task_id=task_id, experiment_name=cfg.experiment_name,
                         max_interactions=cfg.max_iter)
        out = TaskOutcome(task_id=task_id)
        task_dir = Path(cfg.output_dir or ".") / f"task_{task_id}"
        task_dir.mkdir(parents=True, exist_ok=True)

        instruction = world.task.instruction
        api_docs = str(world.task.api_docs) if cfg.api_docs_mode == "full" else ""
        system = cfg.agent_prompt or DEFAULT_AGENT_PROMPT

        history: list[Turn] = []
        sessions: list[list[dict]] = [[]]
        summary = ""
        total_in = total_out = 0
        try:
            for step in range(1, cfg.max_iter + 1):
                ctx = build_context(instruction, api_docs, history, summary, cfg)
                reply = self.llm.chat(ctx, system=system, temperature=cfg.temperature)
                sessions[-1].extend([{"role": "user", "content": ctx},
                                     {"role": "assistant", "content": reply}])
                total_in += len(ctx) // 4
                total_out += len(reply) // 4

                code = extract_code(reply)
                obs = world.execute(code)
                history.append(Turn(index=step, action=code, observation=obs))
                out.steps = step

                if world.task_completed():
                    break
                if step >= cfg.max_iter:
                    out.hit_cap = True
                    break

                # Compaction boundary: only the compressible span is handed over.
                span = sum(len(t.action) + len(t.observation) for t in history)
                if policy is not None and span > cfg.history_budget * 4:
                    req = CompressionRequest(
                        task=instruction, history=history, prev_summary=summary,
                        budget=cfg.history_budget, is_first=(summary == ""))
                    res = policy.compress(req)
                    summary = res.text
                    history = []            # replaced by the summary
                    sessions.append([])     # new session, mirroring the harness
                    out.compactions += 1
        except Exception as e:              # noqa: BLE001 - recorded, not swallowed
            out.error = f"{type(e).__name__}: {e}"
        finally:
            self._write(task_dir, history, sessions, total_in, total_out, out, policy, cfg)
            try:
                world.close()
            except Exception:
                pass
        return out

    @staticmethod
    def _write(task_dir: Path, history: list[Turn], sessions: list[list[dict]],
               total_in: int, total_out: int, out: TaskOutcome,
               policy: CompressionPolicy | None, cfg: EngineConfig) -> None:
        (task_dir / "env_history.json").write_text(json.dumps(
            [{"step": t.index, "action": t.action, "output": t.observation,
              "reward": 0.0, "done": False} for t in history], indent=2))
        (task_dir / "llm_history.json").write_text(json.dumps(sessions, indent=2))
        (task_dir / "token_usage_and_cost.json").write_text(json.dumps(
            {"total_input_tokens": total_in, "total_output_tokens": total_out,
             "total_requests": sum(len(s) // 2 for s in sessions),
             "model_name": cfg.model}, indent=2))
        (task_dir / "results.json").write_text(json.dumps(
            {"iterations": out.steps, "task_id": out.task_id,
             "info": {"success": None,
                      "reason": "max_interactions" if out.hit_cap else "finished"},
             "compactions": out.compactions, "error": out.error,
             "config": {"max_iter": cfg.max_iter, "model": cfg.model,
                        "history_budget": cfg.history_budget,
                        "api_docs_mode": cfg.api_docs_mode,
                        "policy": getattr(policy, "name", None),
                        "experiment_name": cfg.experiment_name}}, indent=2))
