"""
Provider-agnostic async LLM token streaming.

Wraps an OpenAI-compatible ``AsyncOpenAI`` client so the inference
provider is a runtime config, not a hardcoded plugin — point it at Groq,
Cerebras, or any other OpenAI-compatible endpoint via env vars:

    LLM_BASE_URL   default: https://api.groq.com/openai/v1
    LLM_API_KEY
    LLM_MODEL      e.g. llama-3.3-70b-versatile, llama3.3-70b, qwen-3-instruct

Tokens are pushed onto an ``asyncio.Queue`` the instant each delta arrives
on the wire — never accumulated into sentences or the full completion —
so downstream consumers (TTS, SSE, benchmarks) can start acting on the
first token immediately.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Optional

from openai import AsyncOpenAI

from latency import StreamLatency

logger = logging.getLogger("nexus.llm.stream")

# Sentinel placed on the queue to signal end-of-stream to consumers.
# An object() identity check is unambiguous even if a real token is falsy.
STREAM_DONE = object()

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: Optional[str]
    model: str

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            base_url=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
            api_key=os.getenv("LLM_API_KEY"),
            model=os.getenv("LLM_MODEL", DEFAULT_MODEL),
        )


def build_client(config: Optional[LLMConfig] = None) -> AsyncOpenAI:
    """Construct the shared AsyncOpenAI client for the configured provider."""
    config = config or LLMConfig.from_env()
    return AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)


async def stream_to_queue(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    queue: "asyncio.Queue[Any]",
    *,
    request_id: Optional[str] = None,
    **create_kwargs: Any,
) -> dict:
    """Stream a chat completion token-by-token onto ``queue``.

    Puts ``STREAM_DONE`` on the queue when the stream ends, whether it
    finished normally or raised, so consumers never have to poll for
    completion. Returns the latency benchmark summary (ttft_ms, tps, ...).
    """
    request_id = request_id or uuid.uuid4().hex[:8]
    bench = StreamLatency(request_id=request_id, model=model)

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            **create_kwargs,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            token = chunk.choices[0].delta.content
            if not token:
                continue
            bench.record_token()
            await queue.put(token)
        return bench.finish()
    finally:
        await queue.put(STREAM_DONE)


async def stream_completion(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    *,
    request_id: Optional[str] = None,
    **create_kwargs: Any,
) -> AsyncIterator[str]:
    """Async-generator convenience wrapper around ``stream_to_queue``.

    For callers that want ``async for token in stream_completion(...)``
    (e.g. an SSE endpoint) instead of owning the queue directly.
    """
    queue: asyncio.Queue = asyncio.Queue()
    producer = asyncio.create_task(
        stream_to_queue(
            client, model, messages, queue, request_id=request_id, **create_kwargs
        )
    )
    try:
        while True:
            item = await queue.get()
            if item is STREAM_DONE:
                break
            yield item
    finally:
        if not producer.done():
            producer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await producer
