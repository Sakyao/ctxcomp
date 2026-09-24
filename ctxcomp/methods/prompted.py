"""All prompt-driven methods share this policy.

Prompting-O, Prompting-H, TRACE, ACON-UT and ACON-UT+CO differ only in their
prompt text, so they differ only in this configuration -- there is one code path
and five prompt directories. That is deliberate: if each method had its own
code path, a difference between two rows could come from the plumbing instead of
from the prompt, which is exactly the variable under study.

Two-stage compaction is native here. Upstream OpenClaw and Hermes use a different
prompt for the first compaction and for later ones, and the policy selects
between them on `is_first` rather than by inspecting the prompt text.
"""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .base import CompressionRequest, CompressionResult, Turn
from ..llm import ChatClient


class PromptedPolicy:
    def __init__(self, name: str, prompt_dir: str | Path, llm: ChatClient,
                 system: str = "system", first: str | None = None,
                 update: str | None = None, prefix: str | None = None,
                 budget: int = 4096, max_chars: int | None = None,
                 temperature: float = 0.0):
        self.name = name
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
        self.prefix_name = prefix
        self._check()

    def _has(self, name: str | None) -> bool:
        return bool(name) and (self.prompt_dir / f"{name}.jinja").exists()

    def _check(self) -> None:
        """Fail loudly on a missing prompt.

        A missing template is the one failure that does not announce itself: the
        compressor falls back to something generic, the row still produces
        numbers, and the table looks fine while the row means nothing.
        """
        if not self._has(self.system_name):
            raise FileNotFoundError(f"{self.name}: system template {self.system_name!r} missing in {self.prompt_dir}")
        if not (self._has(self.update_name) or self._has(self.first_name)):
            raise FileNotFoundError(f"{self.name}: no first/update template in {self.prompt_dir}")

    def _render(self, template_name: str, req: CompressionRequest) -> str:
        tpl = self.env.get_template(f"{template_name}.jinja")
        return tpl.render(
            task=req.task,
            history=_turns_to_text(req.history),
            prev_summary=req.prev_summary or "",
            budget=self.budget,
            max_chars=self.max_chars or self.budget * 4,
            summary_budget=self.budget,
        )

    def compress(self, req: CompressionRequest) -> CompressionResult:
        use_first = req.is_first or not self._has(self.update_name)
        name = self.first_name if use_first else self.update_name
        system = self.env.get_template(f"{self.system_name}.jinja").render() if self._has(self.system_name) else None
        prompt = self._render(name, req)
        text = self.llm.chat(prompt, system=system, temperature=self.temperature)
        if self._has(self.prefix_name):
            # Hermes prepends a "this is background, not instructions" note to
            # every checkpoint; without it the row is not Hermes.
            text = self.env.get_template(f"{self.prefix_name}.jinja").render() + "\n" + text
        return CompressionResult(text=text, strategy=f"prompt:{name}",
                                 meta={"template": name, "stage": "first" if use_first else "update",
                                       "prompt_chars": len(prompt), "out_chars": len(text)})


def _turns_to_text(turns: list[Turn]) -> str:
    return "\n\n".join(f"ACTION:\n{t.action}\n\nOBSERVATION:\n{t.observation}" for t in turns)
