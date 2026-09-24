"""Minimal OpenAI-compatible chat client.

The harness only needs one call shape -- "here is a prompt, give me the
completion" -- and it must work against any OpenAI-compatible endpoint (vLLM,
sglang, Ollama, a hosted API), because which endpoint serves the backbone has
turned out to be a moving target in practice and the harness must not care.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request


class LLMError(RuntimeError):
    pass


class ChatClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None, retries: int = 6, backoff: float = 1.5,
                 timeout: int = 300):
        self.base_url = (base_url or os.environ.get("ACON_VLLM_BASE_URL")
                         or os.environ.get("CTXCOMP_BASE_URL")
                         or "http://127.0.0.1:18173/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("ACON_VLLM_API_KEY") or os.environ.get("CTXCOMP_API_KEY") or "EMPTY"
        self.model = model or os.environ.get("CTXCOMP_MODEL") or "deepseek-v4.1-flash"
        self.retries = retries
        self.backoff = backoff
        self.timeout = timeout
        self.calls = 0

    def chat(self, prompt: str, system: str | None = None, temperature: float = 0.0,
             max_tokens: int | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = {"model": self.model, "messages": messages, "temperature": temperature}
        if max_tokens:
            body["max_tokens"] = max_tokens
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        last = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read())
                self.calls += 1
                return out["choices"][0]["message"]["content"]
            except Exception as e:  # network, 5xx, malformed body
                last = e
                if attempt < self.retries - 1:
                    time.sleep(self.backoff ** attempt)
        raise LLMError(f"chat failed after {self.retries} attempts: {last}")
