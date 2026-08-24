"""
Manual TTFT/TPS benchmark against the configured LLM provider.

Usage:
    python bench_llm.py "What is the capital of France?"
    python bench_llm.py --runs 5 "Summarize the theory of relativity in one sentence."

Reads LLM_BASE_URL / LLM_API_KEY / LLM_MODEL from the environment (see
llm_stream.LLMConfig) and streams a single completion per run, printing
tokens as they arrive plus the TTFT/TPS summary logged by StreamLatency.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from llm_stream import LLMConfig, build_client, stream_completion

logging.basicConfig(level=logging.INFO, format="%(message)s")


async def run_once(prompt: str, config: LLMConfig, run_id: int) -> dict:
    client = build_client(config)
    messages = [{"role": "user", "content": prompt}]

    print(f"\n--- run {run_id} ({config.model} @ {config.base_url}) ---")
    async for token in stream_completion(
        client, config.model, messages, request_id=f"bench-{run_id}"
    ):
        print(token, end="", flush=True)
    print()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="User prompt to send")
    parser.add_argument("--runs", type=int, default=1, help="Number of sequential runs")
    args = parser.parse_args()

    config = LLMConfig.from_env()
    if not config.api_key:
        print("LLM_API_KEY is not set — aborting.", file=sys.stderr)
        raise SystemExit(1)

    for i in range(1, args.runs + 1):
        await run_once(args.prompt, config, i)


if __name__ == "__main__":
    asyncio.run(main())
