#!/usr/bin/env python3
"""
The LLMLingua HTTP shim that ACON's code expects but does not ship.

`src/productive_agents/ctxopt/base.py:213` never imports the llmlingua package.
It calls an out-of-repo service:

    def llmlingua_compress_context(self, prompt, ratio=0.33):
        r = requests.post("http://localhost:9999/compress",
                          json={"prompt": prompt, "rate": ratio})
        r.raise_for_status()
        return r.json()["compressed_prompt"]

So to reproduce the LLMLingua rows of Tables 1 & 2 you must stand that service up.
This file is that service, using only the standard library for the HTTP layer.

Callers inside the repo:
    history_optimizer.process() -> llmlingua_compress_context(history, ratio=0.2)
    obs_optimizer.process()     -> llmlingua_compress_context(observation, ratio=0.3)
ACON's paper text says "keep rate 30%"; the two ratios above are hard-coded in
the repo, so we honour whatever the caller sends and do not second-guess it.

Install (CPU-only torch keeps this ~1.5 GB instead of ~5 GB):
    python -m venv /path/to/venv
    /path/to/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
    /path/to/venv/bin/pip install llmlingua
    /path/to/venv/bin/pip install sentencepiece  # not always required

Run:
    HF_ENDPOINT=https://hf-mirror.com \
    /path/to/venv/bin/python experiments/repro/llmlingua_server.py --port 9999

Check:
    curl -s localhost:9999/health
    curl -s -X POST localhost:9999/compress \
         -H 'Content-Type: application/json' \
         -d '{"prompt":"a very long context ...","rate":0.3}'
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOGGER = logging.getLogger("llmlingua-server")

COMPRESSOR = None
COMPRESSOR_LOCK = threading.Lock()
ARGS = None

# LLMLingua-2 needs the model trained for it; the v1 default ('NousResearch/Llama-2-7b-hf')
# is 13 GB and needs a GPU. We default to LLMLingua-2 (xlm-roberta-large, ~2 GB, runs on CPU).
DEFAULT_LLMLINGUA2_MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
DEFAULT_V1_MODEL = "NousResearch/Llama-2-7b-hf"


def get_compressor():
    global COMPRESSOR
    if COMPRESSOR is None:
        with COMPRESSOR_LOCK:
            if COMPRESSOR is None:
                from llmlingua import PromptCompressor
                if ARGS.llmlingua2:
                    LOGGER.info("loading LLMLingua-2 compressor: %s (device=%s)",
                                ARGS.model, ARGS.device)
                    COMPRESSOR = PromptCompressor(
                        model_name=ARGS.model,
                        use_llmlingua2=True,
                        device_map=ARGS.device,
                    )
                else:
                    LOGGER.info("loading LLMLingua v1 compressor: %s", ARGS.model)
                    COMPRESSOR = PromptCompressor(
                        model_name=ARGS.model,
                        device_map=ARGS.device,
                    )
                LOGGER.info("compressor ready")
    return COMPRESSOR


def compress(prompt: str, rate: float) -> str:
    c = get_compressor()
    if ARGS.llmlingua2:
        out = c.compress_prompt(
            prompt,
            rate=rate,
            force_tokens=["\n", ".", "?", "!"],
            drop_consecutive=True,
        )
    else:
        out = c.compress_prompt(
            prompt,
            rate=rate,
            condition_in_question="none",
            reorder_context="sort",
        )
    return out.get("compressed_prompt", "")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *a):
        LOGGER.debug(fmt, *a)

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", ""):
            self._send(200, {"status": "ok", "model": ARGS.model,
                             "llmlingua2": ARGS.llmlingua2, "loaded": COMPRESSOR is not None})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/compress":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self._send(400, {"error": f"bad json: {e}"})
            return

        prompt = req.get("prompt")
        rate = req.get("rate", req.get("ratio", 0.33))
        if not isinstance(prompt, str) or not prompt:
            self._send(400, {"error": "missing 'prompt'"})
            return
        try:
            rate = float(rate)
        except Exception:
            rate = 0.33

        try:
            compressed = compress(prompt, rate)
        except Exception as e:
            LOGGER.exception("compression failed")
            self._send(500, {"error": f"{type(e).__name__}: {e}"})
            return

        LOGGER.info("compressed %d -> %d chars (rate=%.2f)",
                    len(prompt), len(compressed), rate)
        self._send(200, {"compressed_prompt": compressed})


def main() -> None:
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default=DEFAULT_LLMLINGUA2_MODEL)
    ap.add_argument("--device", default="cpu", help="cpu | cuda | cuda:0 ...")
    ap.add_argument("--llmlingua2", dest="llmlingua2", action="store_true", default=True)
    ap.add_argument("--v1", dest="llmlingua2", action="store_false",
                    help="use the original perplexity-based LLMLingua instead")
    ap.add_argument("--preload", action="store_true", help="load the model before serving")
    ap.add_argument("-v", "--verbose", action="store_true")
    ARGS = ap.parse_args()

    if not ARGS.llmlingua2 and ARGS.model == DEFAULT_LLMLINGUA2_MODEL:
        ARGS.model = DEFAULT_V1_MODEL

    logging.basicConfig(level=logging.DEBUG if ARGS.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if ARGS.preload:
        get_compressor()

    srv = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    LOGGER.info("listening on http://%s:%d  (model=%s, llmlingua2=%s)",
                ARGS.host, ARGS.port, ARGS.model, ARGS.llmlingua2)
    LOGGER.info("POST /compress {\"prompt\": str, \"rate\": float} -> {\"compressed_prompt\": str}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("shutting down")


if __name__ == "__main__":
    main()
