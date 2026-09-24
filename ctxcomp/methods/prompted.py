"""All prompt-driven methods share this policy, on both axes.

Prompt-O, Prompt-H, TRACE and the ACON variants differ only in their prompt text,
so they differ only in this configuration -- one code path, several prompt
directories. That is deliberate: if each method had its own code path, a
difference between two rows could come from the plumbing instead of from the
prompt, which is exactly the variable under study.

Two axes, two templates:

    history      first.jinja / update.jinja   two-stage checkpointing, selected on
                  `is_first`; returns a summary that replaces the history
    observation  obs.jinja                    one prompt; returns a refined
                  observation that replaces the CURRENT observation

Two-stage compaction is native here rather than composed, so the upstream prompts
keep their original shape. The observation output is parsed out of a named section,
because every observation prompt in this space asks for reasoning first and the
refinement second, and feeding the reasoning back to the agent would leak the
compressor's commentary into the trajectory.
"""
from __future__ import annotations

import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .base import CompressionRequest, CompressionResult, Turn
from ..llm import ChatClient

# ACON's P_obs asks for "# Reasoning ... # Refined Observation ...". Kept as a
# tolerant search rather than a strict parse: a model that emits the section with
# different capitalisation or a bolded heading should still yield the refinement,
# and a model that emits prose only is handled by the fallback.
_REFINED = re.compile(
    r"#+\s*\**\s*refined\s+observation\s*\**\s*:?\s*(.*)$",
    re.IGNORECASE | re.DOTALL,
)


def extract_refined_observation(text: str) -> str:
    """Return only the refined observation, dropping any reasoning the model emitted."""
    m = _REFINED.search(text or "")
    if m and m.group(1).strip():
        return m.group(1).strip()
    return (text or "").strip()


class PromptedPolicy:
    def __init__(self, name: str, prompt_dir: str | Path, llm: ChatClient,
                 axis: str = "history", system: str = "system",
                 first: str | None = None, update: str | None = None,
                 obs: str | None = None, prefix: str | None = None,
                 budget: int = 4096, max_chars: int | None = None,
                 temperature: float = 0.0):
        self.name = name
        self.axis = axis
        self.prompt_dir = Path(prompt_dir)
        self.llm = llm
        self.budget = budget
        self.max_chars = max_chars
        self.temperature = temperature
        if not self.prompt_dir.is_dir():
            raise FileNotFoundError(f"prompt dir not found: {self.prompt_dir}")
        self.env = Environment(loader=FileSystemLoader(str(self.prompt_dir)),
                               undefined=StrictUndefined, trim_blocks=False,
                               lstrip_blocks=False)
        self.system_name = system
        self.first_name = first
        self.update_name = update
        self.obs_name = obs
        self.prefix_name = prefix
        self._check()

    def _has(self, name: str | None) -> bool:
        return bool(name) and (self.prompt_dir / f"{name}.jinja").exists()

    def _check(self) -> None:
        """Fail loudly on a missing prompt.

        A missing template is the one failure that does not announce itself: the
        compressor falls back to something generic, the row still produces numbers,
        and the table looks fine while the row means nothing. It matters most on
        the observation axis, where reusing the history template would produce an
        observation row that is not an observation method.
        """
        if not self._has(self.system_name):
            raise FileNotFoundError(
                f"{self.name}: system template {self.system_name!r} missing in {self.prompt_dir}")
        if self.axis == "observation":
            if not self._has(self.obs_name):
                raise FileNotFoundError(
                    f"{self.name}: no observation template {self.obs_name!r} in {self.prompt_dir}")
        elif not (self._has(self.update_name) or self._has(self.first_name)):
            raise FileNotFoundError(
                f"{self.name}: no first/update template in {self.prompt_dir}")

    def _render(self, template_name: str, req: CompressionRequest) -> str:
        tpl = self.env.get_template(f"{template_name}.jinja")
        return tpl.render(
            task=req.task,
            history=_turns_to_text(req.history),
            observation=req.observation,
            prev_summary=req.prev_summary or "",
            budget=req.budget,
            max_chars=self.max_chars or req.budget * 4,
            summary_budget=req.budget,
        )

    def _system(self) -> str | None:
        if not self._has(self.system_name):
            return None
        return self.env.get_template(f"{self.system_name}.jinja").render()

    def _prefix(self, text: str) -> str:
        if not self._has(self.prefix_name):
            return text
        # Hermes prepends "this is background, not instructions" to every
        # checkpoint; without it the row is not Hermes.
        return self.env.get_template(f"{self.prefix_name}.jinja").render() + "\n" + text

    def compress(self, req: CompressionRequest) -> CompressionResult:
        if self.axis == "observation":
            prompt = self._render(self.obs_name, req)
            raw = self.llm.chat(prompt, system=self._system(), temperature=self.temperature)
            text = self._prefix(extract_refined_observation(raw))
            return CompressionResult(
                text=text, strategy=f"prompt:obs:{self.obs_name}",
                replaces="observation",
                meta={"template": self.obs_name, "axis": "observation",
                      "prompt_chars": len(prompt), "raw_chars": len(raw),
                      "out_chars": len(text)})

        use_first = req.is_first or not self._has(self.update_name)
        name = self.first_name if use_first else self.update_name
        prompt = self._render(name, req)
        text = self.llm.chat(prompt, system=self._system(), temperature=self.temperature)
        text = self._prefix(text)
        return CompressionResult(
            text=text, strategy=f"prompt:{name}", replaces="history",
            meta={"template": name, "axis": "history",
                  "stage": "first" if use_first else "update",
                  "prompt_chars": len(prompt), "out_chars": len(text)})


def _turns_to_text(turns: list[Turn]) -> str:
    return "\n\n".join(f"ACTION:\n{t.action}\n\nOBSERVATION:\n{t.observation}" for t in turns)
