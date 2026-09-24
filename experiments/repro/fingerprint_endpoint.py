#!/usr/bin/env python
"""Fingerprint the inference endpoint so run results can be attributed to a model.

Why
---
The reproduction talks to a shared sglang server whose `/v1/models` listing is
not trustworthy: it echoes back whatever `model` name the client sends, so the
`model` field of a response says nothing about the weights behind it. We also
have direct evidence the server was reconfigured at least once -- the listing
reported `deepseek-v4.1-flash` at 16:52 and `deepseek-v4-flash` later the same
day -- and AppWorld accuracy swung from 82.1 to 32-42 across that window.

`/get_server_info` exposes the full ServerArgs, which gives us several
independent identity signals:

  * `served_model_name`  -- what the operator configured; changes only on restart
  * `random_seed`        -- sglang picks this at startup; a RESTART DETECTOR
  * `max_total_num_tokens`, `tp_size`, `attention_backend` -- server config
  * `version`            -- sglang build

CAREFUL: `served_model_name` is only a label typed by whoever launched the
server, and this deployment demonstrably mislabels it -- on 2026-09-22 the two
endpoints reported `deepseek-v4.1-flash` and `deepseek-v4-flash` while both
mounted `/home/ame/model/DeepSeek-V4-Flash-0731`, i.e. the SAME weights. The
OpenAI-style `model` field in a response is worth even less: the server echoes
back whatever string the client sent, without validating it.

So the authoritative identity is the *weight directory actually mounted*, which
we read straight out of `/proc/<pid>/mountinfo` where the sglang processes run
(`weight_mount_source` below). That cannot be faked by a launch flag. Behavioural
probes are recorded as a secondary check:

  * tokenizer fingerprint -- `usage.prompt_tokens` for a fixed text. Different
    tokenizer => different count, and it needs no extra endpoint access.
  * greedy fingerprint    -- sha256 of the completion at temperature 0 for fixed
    prompts. Different weights => different text (usually).
  * logprob signature     -- top-5 logprobs of the first generated token. This is
    the most sensitive of the three: it moves even when the greedy text does not.

Usage
-----
    python fingerprint_endpoint.py                 # probe + append a record
    python fingerprint_endpoint.py --note "before rerun"
    python fingerprint_endpoint.py --compare       # table of all records
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import requests

DEFAULT_BASE = os.environ.get("ACON_VLLM_BASE_URL", "http://192.168.1.13:18173/v1")
DEFAULT_KEY = os.environ.get("ACON_VLLM_API_KEY", "")
DEFAULT_OUT = Path("/z5s/morph/home/sjk/Agent/datasets/logs/endpoint_fingerprint.jsonl")

# Fixed probes. Keep these stable: changing them invalidates comparability with
# every earlier record. Short enough to be cheap, long enough to be sensitive.
TOKENIZER_PROBE = (
    "Reset friends on venmo to be the same as my friends in my phone. "
    "Befriend and unfriend as needed."
)
GREEDY_PROBES = [
    "Reply with exactly one word: the capital of France.",
    "Complete this Python identifier with one token: apis.supervisor.complete_task(answ",
    "Count from 1 to 5, comma separated, no spaces.",
]
LOGPROB_PROBE = "The supervisor API call to finish a task is apis.supervisor."


def _get(base: str, key: str, path: str, timeout: int = 20):
    r = requests.get(base.replace("/v1", "") + path,
                     headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
    if r.status_code != 200:
        return None
    try:
        return r.json()
    except Exception:
        return None


def _post(base: str, key: str, path: str, body: dict, timeout: int = 120):
    r = requests.post(base + path,
                      headers={"Authorization": f"Bearer {key}",
                               "Content-Type": "application/json"},
                      json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


def probe_weight_mounts() -> dict:
    """Authoritative identity: which host directory each sglang server mounts as /models.

    Every field the HTTP API exposes about "which model" is either a launch-flag
    label (`served_model_name`) or an echo of whatever the client sent (`model` in
    a response). The container's mount table is not: it shows the actual weights
    the process mapped, and no launch flag can fake it.

    Format of /proc/<pid>/mountinfo is
        36 35 98:0 <root> <mount point> <options> ... - <fstype> <source> <super opts>
    so for the /models entry the host directory is field 4.
    """
    found = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="replace")
            if "launch_server" not in cmd:
                continue
            info = open(f"/proc/{pid}/mountinfo").read().splitlines()
        except OSError:
            continue                        # not ours, or already gone

        served_m = re.search(r"--served-model-name\s+(\S+)", cmd)
        served = served_m.group(1) if served_m else ""
        port_m = re.search(r"--port\s+(\d+)", cmd)
        port = port_m.group(1) if port_m else pid
        for line in info:
            parts = line.split()
            if len(parts) >= 5 and parts[4] == "/models":
                found[port] = {
                    "pid": pid,
                    "served_model_name": served,
                    "weight_dir": parts[3],
                }
    return found


def probe(base: str, key: str) -> dict:
    rec: dict = {"ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    root = base.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]

    # --- server identity -------------------------------------------------
    models = _get(base, key, "/v1/models")
    if models and models.get("data"):
        rec["models_id"] = [m.get("id") for m in models["data"]]
        rec["models_root"] = [m.get("root") for m in models["data"]]

    si = _get(base, key, "/get_server_info")
    if si:
        for k in ("served_model_name", "version", "random_seed", "max_total_num_tokens",
                  "tp_size", "dp_size", "attention_backend", "dtype",
                  "enable_deterministic_inference", "model_path", "tokenizer_path"):
            if k in si:
                rec[k] = si[k]

    # --- authoritative identity: the weights actually mounted -------------
    # `served_model_name` above is only a label; this is the directory the server
    # mapped as /models, read from the process's own mount table.
    mounts = probe_weight_mounts()
    if mounts:
        rec["weight_mounts_all"] = mounts
        port_m = re.search(r":(\d+)/", base) or re.search(r":(\d+)$", base)
        if port_m and port_m.group(1) in mounts:
            rec["weight_dir"] = mounts[port_m.group(1)].get("weight_dir")
            rec["weight_dir_pid"] = mounts[port_m.group(1)].get("pid")
        else:
            dirs = sorted({v.get("weight_dir") for v in mounts.values() if v.get("weight_dir")})
            rec["weight_dir"] = dirs[0] if len(dirs) == 1 else None

    # --- weights fingerprints (behavioural) ------------------------------
    try:
        out = _post(base, key, "/chat/completions", {
            "model": rec.get("served_model_name") or "default",
            "messages": [{"role": "user", "content": TOKENIZER_PROBE}],
            "max_tokens": 1, "temperature": 0,
        })
        rec["tokenizer_probe_prompt_tokens"] = (out.get("usage") or {}).get("prompt_tokens")
    except Exception as e:
        rec["tokenizer_probe_error"] = str(e)[:120]

    greedy_hashes = []
    for p in GREEDY_PROBES:
        try:
            out = _post(base, key, "/chat/completions", {
                "model": rec.get("served_model_name") or "default",
                "messages": [{"role": "user", "content": p}],
                "max_tokens": 24, "temperature": 0, "seed": 42,
            })
            txt = out["choices"][0]["message"]["content"] or ""
            greedy_hashes.append(hashlib.sha256(txt.encode()).hexdigest()[:16])
        except Exception as e:
            greedy_hashes.append("ERR:" + str(e)[:40])
    rec["greedy_sha256_16"] = greedy_hashes

    try:
        out = _post(base, key, "/chat/completions", {
            "model": rec.get("served_model_name") or "default",
            "messages": [{"role": "user", "content": LOGPROB_PROBE}],
            "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 5,
        })
        lp = (out["choices"][0].get("logprobs") or {}).get("content") or []
        if lp:
            rec["logprob_top5"] = [
                [t.get("token"), round(t.get("logprob", 0.0), 4)]
                for t in (lp[0].get("top_logprobs") or [])[:5]
            ]
    except Exception as e:
        rec["logprob_error"] = str(e)[:120]

    return rec


def summarize(rec: dict) -> str:
    return "\n".join([
        f"  WEIGHT DIR (authoritative) : {rec.get('weight_dir')}",
        f"  served_model_name (label)  : {rec.get('served_model_name')}",
        f"  /v1/models id (label)      : {rec.get('models_id')}",
        f"  sglang version            : {rec.get('version')}",
        f"  random_seed (restart det.) : {rec.get('random_seed')}",
        f"  max_total_num_tokens      : {rec.get('max_total_num_tokens')}   tp={rec.get('tp_size')}",
        f"  attention_backend         : {rec.get('attention_backend')}",
        f"  deterministic_inference   : {rec.get('enable_deterministic_inference')}",
        f"  tokenizer probe tokens    : {rec.get('tokenizer_probe_prompt_tokens')}",
        f"  greedy sha256 (3 probes)  : {rec.get('greedy_sha256_16')}",
        f"  logprob top5              : {rec.get('logprob_top5')}",
    ])


IDENTITY_KEYS = ["weight_dir", "served_model_name", "models_id", "version",
                 "random_seed", "max_total_num_tokens", "tp_size",
                 "tokenizer_probe_prompt_tokens", "greedy_sha256_16", "logprob_top5"]


def compare(path: Path) -> None:
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not recs:
        print("no records")
        return
    print(f"{len(recs)} record(s) in {path}\n")
    for i, r in enumerate(recs):
        print(f"--- record {i}  ts={r.get('ts')}  note={r.get('note','')}")
        print(summarize(r))
        print()
    if len(recs) > 1:
        print("########## field-by-field changes ##########")
        for k in IDENTITY_KEYS:
            vals = [json.dumps(r.get(k), ensure_ascii=False) for r in recs]
            mark = "  <-- CHANGED" if len(set(vals)) > 1 else ""
            print(f"  {k:30} distinct={len(set(vals))}{mark}")
            if mark:
                for r, v in zip(recs, vals):
                    print(f"        {r.get('ts')}  {v[:110]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--api-key", default=DEFAULT_KEY)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--note", default="")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    path = Path(args.out)
    if args.compare:
        compare(path)
        return 0

    rec = probe(args.base, args.api_key)
    rec["note"] = args.note
    rec["base"] = args.base
    print(f"endpoint fingerprint @ {args.base}")
    print(summarize(rec))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nappended -> {path}")

    # Warn loudly if this probe differs from the previous one: a change means any
    # comparison between runs on either side of it is confounded.
    prev = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if len(prev) > 1:
        a, b = prev[-2], prev[-1]
        diffs = [k for k in IDENTITY_KEYS
                 if json.dumps(a.get(k)) != json.dumps(b.get(k))]
        if diffs:
            print("\n[!] ENDPOINT CHANGED since the previous probe:")
            for k in diffs:
                print(f"      {k}: {json.dumps(a.get(k))[:70]}  ->  {json.dumps(b.get(k))[:70]}")
            print("    Runs on either side of this are not comparable.")
        else:
            print("\n[ok] identical to the previous probe")
    return 0


if __name__ == "__main__":
    sys.exit(main())
