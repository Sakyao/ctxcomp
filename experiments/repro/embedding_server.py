#!/usr/bin/env python3
"""
Local OpenAI-compatible embeddings service, needed by the Retrieval baseline.

Why this exists
---------------
`HistoryRetriever` (src/productive_agents/ctxopt/history_optimizer.py:365-368) does:

    from productive_agents.subtrate_api import AzureOpenAIEmbeddings
    self.embedding_model = AzureOpenAIEmbeddings(model_name="text-embedding-3-large")

i.e. it hard-wires an Azure OpenAI embeddings endpoint. On this machine the
local model server (ACON_VLLM_BASE_URL) answers:

    400 "This model does not appear to be an embedding model by default.
         Please add `--is-embedding` when launching the server"

so the Retrieval rows of the tables cannot run. This service provides a local
replacement; `AzureOpenAIEmbeddings` routes to it when ACON_LOCAL_EMBEDDING_URL
is set (see the small hook in src/productive_agents/subtrate_api.py).

Deviation from the paper: ACON uses text-embedding-3-large. This server defaults
to a local sentence-transformers model instead, because no embedding model is
being served here and all GPUs are occupied. Retrieval only compares cosine
similarities between history turns, so the ranking is what matters; the
substitution should be recorded as a caveat when reporting the Retrieval row.

Install (CPU only):
    /path/to/venv/bin/pip install sentence-transformers

Run:
    HF_ENDPOINT=https://hf-mirror.com \
    /path/to/venv/bin/python experiments/repro/embedding_server.py --port 9200

Then, before running Retrieval:
    export ACON_LOCAL_EMBEDDING_URL=http://127.0.0.1:9200/v1

Check:
    curl -s -X POST localhost:9200/v1/embeddings \
         -H 'Content-Type: application/json' \
         -d '{"model":"bge-large-en-v1.5","input":["hello","world"]}'
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOGGER = logging.getLogger("embedding-server")

MODEL = None
MODEL_LOCK = threading.Lock()
ARGS = None

DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"


def get_model():
    global MODEL
    if MODEL is None:
        with MODEL_LOCK:
            if MODEL is None:
                from sentence_transformers import SentenceTransformer
                LOGGER.info("loading %s (device=%s)", ARGS.model, ARGS.device)
                MODEL = SentenceTransformer(ARGS.model, device=ARGS.device)
                LOGGER.info("ready, dim=%d", MODEL.get_sentence_embedding_dimension())
    return MODEL


def embed(texts: list[str]) -> list[list[float]]:
    m = get_model()
    vecs = m.encode(texts, batch_size=ARGS.batch_size, normalize_embeddings=True,
                    show_progress_bar=False)
    return [v.tolist() for v in vecs]


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
        if self.path.rstrip("/") in ("/health", "/v1/models", "/v1"):
            self._send(200, {"status": "ok", "model": ARGS.model})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/embeddings":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self._send(400, {"error": f"bad json: {e}"})
            return

        inp = req.get("input")
        if isinstance(inp, str):
            inp = [inp]
        if not isinstance(inp, list) or not inp:
            self._send(400, {"error": "missing 'input'"})
            return
        inp = [str(x) for x in inp]

        try:
            vecs = embed(inp)
        except Exception as e:
            LOGGER.exception("embedding failed")
            self._send(500, {"error": f"{type(e).__name__}: {e}"})
            return

        LOGGER.info("embedded %d texts -> dim %d", len(inp), len(vecs[0]) if vecs else 0)
        self._send(200, {
            "object": "list",
            "model": req.get("model", ARGS.model),
            "data": [{"object": "embedding", "index": i, "embedding": v}
                     for i, v in enumerate(vecs)],
            "usage": {"prompt_tokens": sum(len(t.split()) for t in inp),
                      "total_tokens": sum(len(t.split()) for t in inp)},
        })


def main() -> None:
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=9200)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cpu", help="cpu | cuda | cuda:0 ...")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--preload", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    ARGS = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if ARGS.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if ARGS.preload:
        get_model()

    srv = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    LOGGER.info("listening on http://%s:%d/v1/embeddings  (model=%s)", ARGS.host, ARGS.port, ARGS.model)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("shutting down")


if __name__ == "__main__":
    main()
