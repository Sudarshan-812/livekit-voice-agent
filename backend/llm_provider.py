"""
Builds the LiveKit AgentSession LLM plugin from the same env-driven
LLMConfig used by the raw streaming client in llm_stream.py, so the live
voice pipeline and the standalone benchmark/SSE paths always target the
same inference provider — no hardcoded provider plugin (e.g. groq.LLM)
baked into the worker.
"""

from __future__ import annotations

from typing import Optional

from livekit.plugins import openai as lk_openai

from llm_stream import LLMConfig


def build_llm(config: Optional[LLMConfig] = None) -> lk_openai.LLM:
    """Return a livekit-agents LLM plugin pointed at LLM_BASE_URL/LLM_MODEL.

    Works with any OpenAI-compatible chat completions endpoint (Groq,
    Cerebras, etc.) since livekit's openai.LLM plugin accepts an arbitrary
    base_url.
    """
    config = config or LLMConfig.from_env()
    return lk_openai.LLM(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
    )
