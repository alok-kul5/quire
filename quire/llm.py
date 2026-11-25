"""Minimal Ollama chat client, plus a deterministic fake used in tests and CI."""

from __future__ import annotations

import re
from typing import Protocol

import httpx

from . import config


class LLM(Protocol):
    name: str

    def chat(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 512) -> str: ...


class OllamaLLM:
    def __init__(self, model: str = config.LLM_MODEL, url: str = config.OLLAMA_URL, timeout: float = 600.0, retries: int = 0):
        self.name = model
        self.retries = retries
        self._suffix = "\n/no_think" if "qwen3" in model else ""  # qwen3 soft switch: no chain-of-thought
        self.url = url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def chat(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        for attempt in range(self.retries + 1):
            try:
                return self._chat(system, user, temperature, max_tokens)
            except (httpx.TimeoutException, httpx.ConnectError):
                if attempt == self.retries:
                    raise
        raise RuntimeError("unreachable")

    def _chat(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        r = self.client.post(
            f"{self.url}/api/chat",
            json={
                "model": self.name,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user + self._suffix}],
                "stream": False,
                "think": False,
                "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 4096, "seed": 7},
            },
        )
        r.raise_for_status()
        text = r.json()["message"]["content"]
        return strip_reasoning(text)

    def available(self) -> bool:
        try:
            tags = self.client.get(f"{self.url}/api/tags", timeout=3).json()
            return any(m["name"].split(":")[0] == self.name.split(":")[0] for m in tags.get("models", []))
        except Exception:
            return False


class FakeLLM:
    """Echoes the first context passage with a citation, or refuses when there is none.

    Lets the whole pipeline (API, UI, eval plumbing) run in CI with no model.
    """

    name = "fake"

    def __init__(self, reply: str | None = None):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def chat(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        self.calls.append((system, user))
        if self.reply is not None:
            return self.reply
        m = re.search(r"\[1\][^\n]*\n(.+?)(?:\n\n|\Z)", user, flags=re.S)
        if not m:
            return "INSUFFICIENT_CONTEXT"
        first = m.group(1).strip().split(". ")[0].rstrip(".")
        return f"{first} [1]."


def strip_reasoning(text: str) -> str:
    """Drop chain-of-thought some local models emit even with thinking disabled."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return re.sub(r"<think>.*", "", text, flags=re.S).strip()
