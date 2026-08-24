"""
Unit tests for the provider-agnostic LLM streaming layer, using a mocked
AsyncOpenAI-shaped client — no network calls, no livekit-agents import.

Run with:
    python -m unittest test_llm_stream -v
"""

from __future__ import annotations

import asyncio
import unittest

from latency import StreamLatency
from llm_stream import STREAM_DONE, stream_completion, stream_to_queue


class _FakeDelta:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.delta = _FakeDelta(content)


class _FakeChunk:
    def __init__(self, content: str | None) -> None:
        self.choices = [_FakeChoice(content)] if content is not None else []


class _FakeStream:
    """Mimics AsyncOpenAI's streamed ChatCompletionChunk iterator."""

    def __init__(self, tokens: list[str], delay: float = 0.0) -> None:
        self._tokens = tokens
        self._delay = delay

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for token in self._tokens:
            if self._delay:
                await asyncio.sleep(self._delay)
            yield _FakeChunk(token)


class _FakeCompletions:
    def __init__(self, tokens: list[str], delay: float = 0.0) -> None:
        self._tokens = tokens
        self._delay = delay

    async def create(self, **kwargs):
        return _FakeStream(self._tokens, self._delay)


class _RaisingCompletions:
    async def create(self, **kwargs):
        raise RuntimeError("upstream provider error")


class _FakeClient:
    def __init__(self, completions) -> None:
        self.chat = _Chat(completions)


class _Chat:
    def __init__(self, completions) -> None:
        self.completions = completions


class StreamToQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_tokens_arrive_in_order(self) -> None:
        tokens = ["Hel", "lo", " world"]
        client = _FakeClient(_FakeCompletions(tokens))
        queue: asyncio.Queue = asyncio.Queue()

        summary = await stream_to_queue(
            client, "test-model", [{"role": "user", "content": "hi"}], queue,
            request_id="t1",
        )

        received = []
        while True:
            item = queue.get_nowait()
            if item is STREAM_DONE:
                break
            received.append(item)

        self.assertEqual(received, tokens)
        self.assertEqual(summary["tokens"], len(tokens))
        self.assertIsNotNone(summary["ttft_ms"])

    async def test_no_sentence_buffering(self) -> None:
        """Tokens must land on the queue as each chunk arrives, not after
        the full completion — proven by observing partial delivery while
        the producer is still running."""
        tokens = ["a", "b", "c"]
        client = _FakeClient(_FakeCompletions(tokens, delay=0.05))
        queue: asyncio.Queue = asyncio.Queue()

        producer = asyncio.create_task(
            stream_to_queue(client, "test-model", [], queue, request_id="t2")
        )

        first = await asyncio.wait_for(queue.get(), timeout=1)
        self.assertEqual(first, "a")
        # Only ~50ms has elapsed (one token delay) — the other two tokens
        # (another ~100ms out) haven't been generated yet, so the producer
        # must still be running. If tokens were buffered until the full
        # response, the producer would already be done here.
        self.assertFalse(producer.done())

        await producer

    async def test_done_sentinel_on_error(self) -> None:
        client = _FakeClient(_RaisingCompletions())
        queue: asyncio.Queue = asyncio.Queue()

        with self.assertRaises(RuntimeError):
            await stream_to_queue(client, "test-model", [], queue)

        item = await asyncio.wait_for(queue.get(), timeout=1)
        self.assertIs(item, STREAM_DONE)

    async def test_empty_deltas_are_skipped(self) -> None:
        tokens = ["hi", "", None, "there"]
        client = _FakeClient(_FakeCompletions(tokens))
        queue: asyncio.Queue = asyncio.Queue()

        await stream_to_queue(client, "test-model", [], queue)

        received = []
        while True:
            item = queue.get_nowait()
            if item is STREAM_DONE:
                break
            received.append(item)
        self.assertEqual(received, ["hi", "there"])


class StreamCompletionGeneratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_yields_tokens_in_order(self) -> None:
        tokens = ["The", " sky", " is", " blue"]
        client = _FakeClient(_FakeCompletions(tokens))

        collected = [
            token async for token in stream_completion(client, "test-model", [])
        ]

        self.assertEqual(collected, tokens)

    async def test_propagates_provider_errors(self) -> None:
        client = _FakeClient(_RaisingCompletions())

        with self.assertRaises(RuntimeError):
            async for _ in stream_completion(client, "test-model", []):
                pass


class StreamLatencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_ttft_and_tps_are_recorded(self) -> None:
        bench = StreamLatency(request_id="r1", model="test-model")

        await asyncio.sleep(0.02)
        bench.record_token()
        await asyncio.sleep(0.02)
        bench.record_token()

        summary = bench.finish()

        self.assertEqual(summary["tokens"], 2)
        self.assertIsNotNone(summary["ttft_ms"])
        self.assertGreater(summary["ttft_ms"], 0)
        self.assertGreater(summary["tps"], 0)

    async def test_no_tokens_yields_null_ttft_and_zero_tps(self) -> None:
        bench = StreamLatency(request_id="r2", model="test-model")
        summary = bench.finish()

        self.assertIsNone(summary["ttft_ms"])
        self.assertEqual(summary["tokens"], 0)
        self.assertEqual(summary["tps"], 0.0)


if __name__ == "__main__":
    unittest.main()
