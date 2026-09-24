"""Reference execution loop: one AppWorld task, one compression policy.

Owned here rather than delegated, because the compaction boundary is the object of
study. The loop applies both axes:

    ask the model -> execute the code -> [observation axis] -> record the turn
                  -> [history axis] -> continue

    observation  o'_t = f(o_t, h_{t-1}; P_obs)   if |o_t| > T_obs   (eq. 4)
                 applied to the observation BEFORE it enters the history, which is
                 what equation (4) means: the refinement is conditioned on the
                 history so far, and the raw observation never reaches the agent.
                 Getting this order wrong -- recording first and refining later --
                 would compress an observation the agent has already read.

    history      h'_t = f(h_t; P_hist)           if |h_t| > T_hist  (eq. 3)
                 applied after the turn is recorded, because the history that trips
                 the threshold includes it.

The policy never sees the environment, the tools or the evaluator; it only sees the
compressible input and returns text. Everything else is frozen, which is what makes
two rows comparable.

Fidelity notes, since they are the difference between this loop and the harness the
baselines were originally measured on:

* `api_docs_mode` decides whether the per-task API documentation is pasted into
  every prompt. It must stay "discover": the reference harness does not paste it,
  and its first user message is ~7.3k characters containing no documentation at all.
  Measured both ways on one task -- 3,700 input tokens per request when the agent
  looks documentation up through `apis.api_docs.*`, versus 108,721 when it is pasted,
  a factor of 30 that decides whether a 1000-step run is feasible.
* Compaction triggers on characters, not tokens, to keep the boundary
  dependency-free. The budget is therefore approximate and is recorded per run.

Artefacts, and who reads them:

    Steps / Peak / Dep   <output_dir>/<task_id>/{env_history,llm_history,
                         token_usage_and_cost,results}.json
    Acc (the evaluator)  <output_dir>/<task_id>/{dbs,logs,...} and, one level up,
                         evaluations/<split>.json

`output_dir` is `<run_dir>/tasks`, so both sets land in the same folder. It has to
be split at the writer level -- the evaluator verifies a task by replaying the
final database state (dbs/) against ground truth and only AppWorld can produce it
-- but it is one directory on disk, which is what matters when reading a result.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..llm import ChatClient
from ..methods.base import CompressionPolicy, CompressionRequest, Turn
from .base import EngineConfig, TaskOutcome

CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.S)

# The agent prompt is not invented here. `prompts/agent/` holds the reference
# harness's own prompt, copied from
#   acon/src/productive_agents/agents/appworld/agent.py  (PROMPT_TEMPLATE)
#   acon/experiments/appworld/prompts/prompt_v1_answerfix.jinja
# and it matters more than it looks: it teaches the agent that API documentation is
# retrieved at runtime rather than supplied, naming the three calls to do it
# (apis.api_docs.show_app_descriptions / show_api_descriptions / show_api_doc).
#
# Measured cost of getting this wrong, on one AppWorld task:
#   with documentation pasted into every prompt   108,721 input tokens per request
#   with the agent looking it up                     ~3,700 input tokens per request
# A factor of 30 that decides whether a 1000-step run is feasible.
AGENT_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts" / "agent"

# Used for steps after the first. The reference harness's first user message is
# 7,333 characters (the full prompt above, with a worked example) but its later
# requests average 1,572 -- the preamble is not resent. So later steps carry a
# compact context plus a reminder of where the documentation lives.
CONTINUATION_REMINDER = """Reminder: you are not given API documentation. Look it up with
  apis.api_docs.show_app_descriptions()
  apis.api_docs.show_api_descriptions(app_name='<app>')
  apis.api_docs.show_api_doc(app_name='<app>', api_name='<api>')
Reply with one Python code block."""


def load_agent_prompt(cfg: "EngineConfig") -> tuple[str, "Template | None"]:
    """(system_message, first_user_template) from the configured prompt JSON.

    The manifest names a prompt file, and until now the engine never read it: it
    used a prompt invented here, which told the agent it would be given API
    documentation. It never was, so the agent guessed signatures and every call
    failed with "Usage of the following APIs ..." -- visible in the smoke run's
    trajectory, 6 steps and 6 execution errors.
    """
    from jinja2 import Template

    spec_path = Path(cfg.prompt_file) if cfg.prompt_file else None
    if not spec_path:
        return "", None
    if not spec_path.is_absolute():
        spec_path = AGENT_PROMPT_DIR / spec_path.name
    if not spec_path.exists():
        spec_path = AGENT_PROMPT_DIR / "prompts_v1_answerfix.json"
    spec = json.loads(spec_path.read_text())
    system = spec.get("system_message", "")
    tpl_name = Path(str(spec.get("main_prompt_template", ""))).name
    tpl_path = AGENT_PROMPT_DIR / tpl_name
    if not tpl_path.exists():
        tpl_path = AGENT_PROMPT_DIR / "prompt_v1_answerfix.jinja"
    return system, Template(tpl_path.read_text())


def extract_code(text: str) -> str:
    m = CODE_BLOCK.search(text or "")
    return (m.group(1) if m else (text or "")).strip()


def build_context(instruction: str, api_docs: str, history: list[Turn],
                  summary: str, cfg: EngineConfig,
                  first_template=None, supervisor=None, is_first: bool = False) -> str:
    if is_first and first_template is not None:
        # The reference prompt verbatim, including its worked example.
        return first_template.render(instruction=instruction, supervisor=supervisor)
    parts = [f"# Task\n{instruction}", CONTINUATION_REMINDER]
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


def span_chars(history: list[Turn]) -> int:
    return sum(len(t.action) + len(t.observation) for t in history)


def _compress(policy: CompressionPolicy, instruction: str, history: list[Turn],
              summary: str, cfg: EngineConfig, observation: str = ""):
    """Call the policy on the axis it declares.

    The axis is read from the policy, never inferred from which threshold tripped:
    a policy that declares `history` must not receive an observation request, even
    if an observation happens to be over budget.
    """
    axis = getattr(policy, "axis", "history")
    budget = cfg.budget_obs if axis == "observation" else cfg.budget_hist
    req = CompressionRequest(task=instruction, axis=axis, history=list(history),
                             observation=observation, prev_summary=summary,
                             budget=budget, is_first=(summary == ""))
    return policy.compress(req)


class AppWorldEngine:
    def __init__(self, llm: ChatClient | None = None):
        self.llm = llm or ChatClient()

    def run_task(self, task_id: str, policy: CompressionPolicy | None,
                 cfg: EngineConfig) -> TaskOutcome:
        from appworld import AppWorld

        world = AppWorld(task_id=task_id, experiment_name=cfg.experiment_name,
                         max_interactions=cfg.max_iter)
        out = TaskOutcome(task_id=task_id)
        # Same folder AppWorld writes dbs/ and logs/ into: one task's whole record,
        # metrics and environment state, side by side. The id is not prefixed with
        # "task_" because AppWorld does not prefix it, and a second naming scheme
        # would put the two halves of a run in different directories again.
        task_dir = Path(cfg.output_dir or ".") / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        instruction = world.task.instruction
        api_docs = str(world.task.api_docs) if cfg.api_docs_mode == "paste-all" else ""
        system, first_template = load_agent_prompt(cfg)
        if cfg.agent_prompt:                       # explicit override wins
            system = cfg.agent_prompt
        supervisor = world.task.supervisor
        axis = getattr(policy, "axis", None) if policy is not None else None

        history: list[Turn] = []          # what the agent currently remembers
        trajectory: list[Turn] = []       # every step that ever happened
        sessions: list[list[dict]] = [[]]
        summary = ""
        total_in = total_out = 0
        try:
            for step in range(1, cfg.max_iter + 1):
                ctx = build_context(instruction, api_docs, history, summary, cfg,
                                    first_template=first_template,
                                    supervisor=supervisor, is_first=(step == 1))
                reply = self.llm.chat(ctx, system=system, temperature=cfg.temperature)
                sessions[-1].extend([{"role": "user", "content": ctx},
                                     {"role": "assistant", "content": reply}])
                total_in += len(ctx) // 4
                total_out += len(reply) // 4

                code = extract_code(reply)
                observation = world.execute(code)

                # observation axis, before the observation enters the history
                if axis == "observation" and len(observation) > cfg.budget_obs * 4:
                    res = _compress(policy, instruction, history, summary, cfg,
                                    observation=observation)
                    observation = self._accept(res, observation, axis)
                    out.compactions_obs += 1

                turn = Turn(index=step, action=code, observation=observation)
                history.append(turn)
                trajectory.append(turn)
                out.steps = step

                if world.task_completed():
                    break
                if step >= cfg.max_iter:
                    out.hit_cap = True
                    break

                # history axis, after the turn is recorded
                if axis == "history" and span_chars(history) > cfg.budget_hist * 4:
                    res = _compress(policy, instruction, history, summary, cfg)
                    summary = self._accept(res, summary, axis)
                    history = []                # replaced by the summary
                    sessions.append([])         # new session, mirroring the harness
                    out.compactions_hist += 1
        except Exception as e:                  # noqa: BLE001 - recorded, not swallowed
            out.error = f"{type(e).__name__}: {e}"
        finally:
            if cfg.save_appworld_artifacts:
                # The evaluator reads tasks/<task_id>/dbs to verify the task against
                # ground truth. AppWorld maintains dbs itself while APIs execute;
                # save_logs adds the api-call and environment-io logs, which are
                # what make a failure diagnosable after the fact.
                try:
                    world.save_logs()
                except Exception as e:          # noqa: BLE001
                    out.meta.setdefault("warnings", []).append(f"save_logs: {e}")
            self._write(task_dir, trajectory, sessions, total_in, total_out, out, policy, cfg)
            try:
                world.close()
            except Exception:
                pass
        return out

    @staticmethod
    def _accept(res, fallback: str, axis: str) -> str:
        """Take the policy's replacement, refusing a mismatch instead of corrupting.

        A policy that returns `replaces: history` while the runner asked for an
        observation has misunderstood the axis. Substituting a summary for an
        observation would look like a working run and produce meaningless numbers.
        """
        if getattr(res, "replaces", axis) != axis:
            raise ValueError(
                f"policy {getattr(res, 'strategy', '?')} returned a "
                f"{getattr(res, 'replaces', '?')!r} replacement while the runner "
                f"asked for {axis!r}")
        return res.text if res.text.strip() else fallback

    @staticmethod
    def _write(task_dir: Path, history: list[Turn], sessions: list[list[dict]],
               total_in: int, total_out: int, out: TaskOutcome,
               policy: CompressionPolicy | None, cfg: EngineConfig) -> None:
        # Write `trajectory`, never the compressible `history`. The two differ once a
        # history-axis policy fires: `history` is cleared and replaced by a summary,
        # so writing it would record only the steps after the last compaction and
        # Steps -- mean environment interactions -- would be undercounted. Measured
        # by the smoke test: 10 steps executed, 3 recorded.
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
             "compactions": {"history": out.compactions_hist,
                             "observation": out.compactions_obs},
             "error": out.error, "warnings": out.meta.get("warnings", []),
             "config": {"max_iter": cfg.max_iter, "model": cfg.model,
                        "budget_hist": cfg.budget_hist,
                        "budget_obs": cfg.budget_obs,
                        "api_docs_mode": cfg.api_docs_mode,
                        "policy": getattr(policy, "name", None),
                        "axis": getattr(policy, "axis", None),
                        "experiment_name": cfg.experiment_name}}, indent=2))
