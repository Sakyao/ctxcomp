#!/usr/bin/env python3
"""Turn methods.yml into a runnable co_config per row.

For each row writes configs/<row>/<row>.yaml plus a prompts/ directory, so the
generated config is self-contained and can be copied or diffed as a unit.

Two things this deliberately refuses to do silently:

1. Emit a row whose prompt cannot be resolved. `_load_prompt_templates` keys
   templates by filename-without-extension, so a config that names a template it
   cannot find falls back to the stock guideline instead of failing -- an
   "ACON-UT" row then becomes indistinguishable from "Prompting" while still
   producing a plausible-looking table.

2. Pretend upstream's two-stage compaction exists. OpenClaw and Hermes each use a
   different prompt for the first compaction and for later ones, but the
   compressor exposes a single `prompt_history_user` and a `prev_summary`
   variable that is the empty string on the first call. The two upstream prompts
   are therefore composed into one template that branches on `prev_summary`;
   COMPOSED_TWO_STAGE records that for the write-up, since it is an adaptation
   rather than a port.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import yaml

SUITE = Path(__file__).resolve().parent
CONFIGS = SUITE / "configs"
COMPOSED_TWO_STAGE: list[str] = []

HISTORY_THRESHOLD = 4096


def load_methods() -> dict:
    return yaml.safe_load((SUITE / "methods.yml").read_text())


def resolve_dir(row: dict) -> Path | None:
    ctx = row.get("ctxopt") or {}
    d = ctx.get("prompt_dir")
    if not d:
        return None
    p = (SUITE / d).resolve() if not Path(d).is_absolute() else Path(d)
    return p


def compose_two_stage(first: str, update: str) -> str:
    """One template that is `first` on the initial call and `update` afterwards.

    `prev_summary` is '' on the first compaction, which is exactly the signal the
    upstream code uses to choose between its two prompts.
    """
    return (
        "{% if prev_summary %}\n"
        + update
        + "\n{% else %}\n"
        + first
        + "\n{% endif %}\n"
    )


def build_row(row: dict, meta: dict) -> dict | None:
    name, kind = row["name"], row["kind"]
    if kind == "none":
        return None  # no compressor at all

    ctx = dict(row.get("ctxopt") or {})
    src = resolve_dir(row)
    if src is None or not src.is_dir():
        print(f"  skip  {name:16} prompt_dir not found: {src}")
        return None

    out_prompts = CONFIGS / name / "prompts"
    out_prompts.mkdir(parents=True, exist_ok=True)

    prompts: dict[str, str] = {}
    # Both axes share the compressor class hierarchy but not the template key. The
    # history optimiser reads `prompt_history_user` and the observation optimiser
    # reads `prompt_user` (obs_optimizer.py:54), while `_load_prompt_templates` keys
    # each template by its filename without the extension. Emitting the history key
    # for an observation row makes the compressor silently fall back to its stock
    # guideline, so the row would carry one method's name and behave like another's.
    #
    # This branch is the whole extent of the local adaptation in this file. Upstream
    # (`experiments/suite/make_configs.py`) hard-codes the history key, so its
    # `kind: obs` path cannot produce a working config; nothing else differs.
    kind_prefix = "history" if kind == "history" else "obs"
    user_key = "prompt_history_user" if kind_prefix == "history" else "prompt_user"
    user_file = "history" if kind_prefix == "history" else "prompt_user"

    # system template
    sysname = ctx.get("system", "system_prompt")
    sysfile = src / f"{sysname}.jinja"
    if not sysfile.exists():
        print(f"  skip  {name:16} missing system template {sysfile.name} in {src}")
        return None
    shutil.copy(sysfile, out_prompts / "system_prompt.jinja")
    prompts["prompt_system"] = "system_prompt"

    # Compressible-input template. The history axis is either a straight copy or a
    # two-stage composition; the observation axis is always a single template,
    # because no upstream project ships a first/update split for observations.
    histname = ctx.get("obs") if kind_prefix == "obs" else ctx.get("history")
    firstname = ctx.get("first")
    if histname and (src / f"{histname}.jinja").exists() and not firstname:
        shutil.copy(src / f"{histname}.jinja", out_prompts / f"{user_file}.jinja")
    elif histname and firstname and (src / f"{firstname}.jinja").exists() and (src / f"{histname}.jinja").exists():
        body = compose_two_stage(
            (src / f"{firstname}.jinja").read_text(),
            (src / f"{histname}.jinja").read_text(),
        )
        (out_prompts / f"{user_file}.jinja").write_text(body)
        COMPOSED_TWO_STAGE.append(name)
    else:
        print(f"  skip  {name:16} missing compressible-input template in {src}")
        return None
    prompts[user_key] = user_file

    # Hermes ships its compaction prefix as a separate artefact; fold it in so the
    # checkpoint actually carries the note the downstream agent is meant to read.
    prefix = src / "prefix.jinja"
    if prefix.exists():
        with_prefix = out_prompts / "prefix.jinja"
        shutil.copy(prefix, with_prefix)

    cfg = {
        "type": kind_prefix,
        "model": meta["model"],
        "compressor_type": "full",
        "prompts": prompts,
        f"{kind_prefix}_summarization_threshold": (
            HISTORY_THRESHOLD if kind_prefix == "history" else 256
        ),
        f"{kind_prefix}_summary_rule": "reset",
        f"{kind_prefix}_prompt_dir": str(out_prompts),
    }
    cfg[f"{kind_prefix}_version"] = 1  # read by memory.py
    if row.get("baseline"):
        cfg["baseline_strategy"] = row["baseline"]
    if row.get("preserve_last_k_turns") is not None:
        cfg["preserve_last_k_turns"] = row["preserve_last_k_turns"]
    if row.get("llmlingua"):
        cfg["use_llmlingua"] = True
        cfg["keep_rate"] = row.get("keep_rate", 0.30)
    return cfg


def main() -> int:
    man = load_methods()
    meta = {"model": man["model"]}
    CONFIGS.mkdir(parents=True, exist_ok=True)
    made, skipped = [], []
    print(f"  methods.yml: {len(man['rows'])} rows, model={meta['model']}")
    for row in man["rows"]:
        cfg = build_row(row, meta)
        if cfg is None:
            skipped.append(row["name"])
            continue
        d = CONFIGS / row["name"]
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{row['name']}.yaml").write_text(
            f"# generated by make_configs.py from methods.yml -- do not edit\n"
            + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
        )
        made.append(row["name"])
        print(f"  build {row['name']:16} -> configs/{row['name']}/{row['name']}.yaml")
    print(f"\n  built {len(made)}: {', '.join(made)}")
    if skipped:
        print(f"  skipped {len(skipped)}: {', '.join(skipped)}")
    if COMPOSED_TWO_STAGE:
        print(f"  two-stage composed into one template: {', '.join(COMPOSED_TWO_STAGE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
