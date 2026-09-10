"""Provider-agnostic LLM client.

The discovery agent must not know or care which model it is talking to. That is
not architectural decoration: the model appears exactly once in this system, in
a run that happens once per capability, and never in the production replay
path. Swapping providers should therefore change one config value and nothing
else -- which is also the honest answer to "why this model?".

xAI (Grok) and OpenAI both speak the same Chat Completions shape, so one
implementation covers them; Anthropic gets its own thin adapter.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str | None = None


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None


class LLMClient(Protocol):
    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
    ) -> LLMResponse: ...

    @property
    def model(self) -> str: ...

    @property
    def supports_vision(self) -> bool: ...


class OpenAICompatClient:
    """Covers xAI/Grok and OpenAI. Uses plain HTTP so no vendor SDK is required.

    Vision is a capability question, not a provider question: if the configured
    model cannot accept images, the discovery agent still functions on the
    accessibility tree alone, which is the primary signal by design. Screenshots
    are a disambiguation aid, not a dependency.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.x.ai/v1",
        vision: bool | None = None,
        env_key: str = "XAI_API_KEY",
    ) -> None:
        self._model = model
        self._base = base_url.rstrip("/")
        self._key = api_key or os.environ.get(env_key) or os.environ.get("OPENAI_API_KEY")
        if not self._key:
            raise RuntimeError(
                f"No API key. Set {env_key} (or pass api_key=). The discovery "
                "run is the one part of this system that cannot be faked."
            )
        self._vision = vision if vision is not None else ("vision" in model or "grok-4" in model)

    @property
    def model(self) -> str:
        return self._model

    @property
    def supports_vision(self) -> bool:
        return self._vision

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        import ssl
        import urllib.error
        import urllib.request

        # python.org builds on macOS ship without a CA bundle, so verification
        # fails against every https endpoint. Use certifi when present rather
        # than disabling verification -- an API key travels on this request.
        try:
            import certifi

            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = ssl.create_default_context()

        body: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "system", "content": system}, *messages],
            "max_tokens": max_tokens,
            "temperature": 0,  # discovery should be as repeatable as a model allows
        }
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
            body["tool_choice"] = "required"

        req = urllib.request.Request(
            f"{self._base}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120, context=ctx) as r:
                payload = json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"LLM call failed {e.code}: {e.read()[:500]!r}") from e

        choice = payload["choices"][0]["message"]
        calls = [
            ToolCall(
                name=tc["function"]["name"],
                arguments=json.loads(tc["function"]["arguments"] or "{}"),
                call_id=tc.get("id"),
            )
            for tc in (choice.get("tool_calls") or [])
        ]
        return LLMResponse(text=choice.get("content") or "", tool_calls=calls, raw=payload)


def build_client(provider: str | None = None, model: str | None = None) -> LLMClient:
    """One switch. Defaults to Grok."""
    provider = (provider or os.environ.get("CUA_LLM_PROVIDER") or "xai").lower()
    if provider in {"xai", "grok"}:
        return OpenAICompatClient(
            model=model or os.environ.get("CUA_LLM_MODEL", "grok-4"),
            base_url="https://api.x.ai/v1",
            env_key="XAI_API_KEY",
        )
    if provider == "ollama":
        # Ollama serves an OpenAI-compatible endpoint, so local models and
        # Ollama Cloud models (the `-cloud` suffixed ones) both arrive through
        # the same client. No API key: the endpoint is on this machine.
        return OpenAICompatClient(
            model=model or os.environ.get("CUA_LLM_MODEL", "qwen3:8b"),
            base_url=os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434/v1"),
            api_key="ollama",  # placeholder; the local server ignores it
            vision=False,
            env_key="OLLAMA_API_KEY",
        )
    if provider in {"nvidia", "nim"}:
        # NVIDIA NIM speaks the OpenAI Chat Completions shape, so it needs no
        # adapter of its own -- which is the point of keeping the discovery
        # agent provider-agnostic.
        return OpenAICompatClient(
            model=model or os.environ.get("CUA_LLM_MODEL", "deepseek-ai/deepseek-v3.1"),
            base_url="https://integrate.api.nvidia.com/v1",
            env_key="NVIDIA_API_KEY",
        )
    if provider == "openai":
        return OpenAICompatClient(
            model=model or "gpt-4o",
            base_url="https://api.openai.com/v1",
            env_key="OPENAI_API_KEY",
        )
    raise ValueError(f"unknown provider {provider!r}")
