"""
Unit tests for the streaming TTS worker, using an in-memory fake
TTSProviderClient — no real WebSocket, no network calls.

Run with:
    python -m unittest test_tts_streamer -v
"""

from __future__ import annotations

import asyncio
import unittest
from typing import Any, Optional

from tts_streamer import (
    SHUTDOWN,
    TURN_END,
    ClauseBuffer,
    TTSProviderClient,
    stream_tts_worker,
)


class _FakeProviderClient(TTSProviderClient):
    """Mimics a persistent provider connection: audio_chunks() yields the
    given events in order, then blocks forever (never ends on its own),
    matching how a real open WebSocket behaves between turns. An Exception
    in the event list is raised instead of yielded, simulating a drop."""

    def __init__(self, audio_events: list[Any]) -> None:
        self.sample_rate = 24000
        self.num_channels = 1
        self.sent_texts: list[str] = []
        self.flush_count = 0
        self.closed = False
        self._audio_events = audio_events

    async def connect(self) -> None:
        pass

    async def send_text(self, text: str) -> None:
        self.sent_texts.append(text)

    async def flush(self) -> None:
        self.flush_count += 1

    async def close(self) -> None:
        self.closed = True

    async def audio_chunks(self):
        for ev in self._audio_events:
            if isinstance(ev, Exception):
                raise ev
            yield ev
        await asyncio.Event().wait()  # persistent connection: never ends on its own


class _ConnectSequence:
    """Injectable `connect` factory returning pre-built fake clients in
    order, clamping to the last one for any extra reconnect attempts."""

    def __init__(self, clients: list[_FakeProviderClient]) -> None:
        self._clients = clients
        self.call_count = 0

    async def __call__(self) -> _FakeProviderClient:
        client = self._clients[min(self.call_count, len(self._clients) - 1)]
        self.call_count += 1
        await client.connect()
        return client


class ClauseBufferTests(unittest.TestCase):
    def test_flushes_on_punctuation(self) -> None:
        buf = ClauseBuffer()
        self.assertEqual(buf.push("Hello"), [])
        self.assertEqual(buf.push(" world"), [])
        self.assertEqual(buf.push(", how are you"), ["Hello world,"])
        self.assertEqual(buf.push("?"), ["how are you?"])

    def test_multiple_boundaries_in_one_token(self) -> None:
        buf = ClauseBuffer()
        self.assertEqual(buf.push("Yes. No! Maybe?"), ["Yes.", "No!", "Maybe?"])

    def test_drain_returns_trailing_partial(self) -> None:
        buf = ClauseBuffer()
        buf.push("no terminator yet")
        self.assertEqual(buf.drain(), "no terminator yet")
        self.assertIsNone(buf.drain())

    def test_reset_discards_partial(self) -> None:
        buf = ClauseBuffer()
        buf.push("discard me")
        buf.reset()
        self.assertIsNone(buf.drain())


class StreamTTSWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_tokens_buffer_into_clauses_and_stream_audio(self) -> None:
        client = _FakeProviderClient([b"aud1", b"aud2"])
        connect = _ConnectSequence([client])
        text_q: asyncio.Queue = asyncio.Queue()
        audio_q: asyncio.Queue = asyncio.Queue()
        interrupt = asyncio.Event()

        worker = asyncio.create_task(stream_tts_worker(text_q, audio_q, interrupt, connect=connect))

        for tok in ["Hello", " world", ", how are you", "?"]:
            await text_q.put(tok)
        await text_q.put(SHUTDOWN)
        await asyncio.wait_for(worker, timeout=2)

        self.assertEqual(client.sent_texts, ["Hello world,", "how are you?"])
        self.assertEqual(client.flush_count, 2)
        self.assertTrue(client.closed)

        audio = []
        while not audio_q.empty():
            audio.append(audio_q.get_nowait())
        self.assertEqual(audio, [b"aud1", b"aud2"])

    async def test_turn_end_flushes_trailing_partial_clause(self) -> None:
        client = _FakeProviderClient([])
        connect = _ConnectSequence([client])
        text_q: asyncio.Queue = asyncio.Queue()
        audio_q: asyncio.Queue = asyncio.Queue()
        interrupt = asyncio.Event()

        worker = asyncio.create_task(stream_tts_worker(text_q, audio_q, interrupt, connect=connect))

        await text_q.put("no punctuation here")
        await text_q.put(TURN_END)
        for _ in range(200):
            if client.flush_count >= 1:
                break
            await asyncio.sleep(0.01)
        else:
            self.fail("TURN_END never triggered a flush")

        self.assertEqual(client.sent_texts, ["no punctuation here"])
        self.assertFalse(client.closed)  # connection stays open across turns

        await text_q.put(SHUTDOWN)
        await asyncio.wait_for(worker, timeout=2)
        self.assertTrue(client.closed)

    async def test_interruption_purges_queues_and_reconnects(self) -> None:
        client1 = _FakeProviderClient([b"partial"])
        client2 = _FakeProviderClient([])
        connect = _ConnectSequence([client1, client2])
        text_q: asyncio.Queue = asyncio.Queue()
        audio_q: asyncio.Queue = asyncio.Queue()
        interrupt = asyncio.Event()

        worker = asyncio.create_task(stream_tts_worker(text_q, audio_q, interrupt, connect=connect))

        chunk = await asyncio.wait_for(audio_q.get(), timeout=2)
        self.assertEqual(chunk, b"partial")  # proves we're mid-stream

        await text_q.put("this token should be discarded")
        interrupt.set()

        for _ in range(200):
            if connect.call_count >= 2:
                break
            await asyncio.sleep(0.01)
        else:
            self.fail("worker did not reconnect after interruption")

        self.assertFalse(interrupt.is_set())
        self.assertTrue(client1.closed)
        self.assertTrue(audio_q.empty())
        self.assertTrue(text_q.empty())

        await text_q.put(SHUTDOWN)
        await asyncio.wait_for(worker, timeout=2)
        self.assertTrue(client2.closed)

    async def test_connection_drop_triggers_reconnect(self) -> None:
        client1 = _FakeProviderClient([b"a1", RuntimeError("socket dropped")])
        client2 = _FakeProviderClient([b"a2"])
        connect = _ConnectSequence([client1, client2])
        text_q: asyncio.Queue = asyncio.Queue()
        audio_q: asyncio.Queue = asyncio.Queue()
        interrupt = asyncio.Event()

        worker = asyncio.create_task(stream_tts_worker(text_q, audio_q, interrupt, connect=connect))

        first = await asyncio.wait_for(audio_q.get(), timeout=2)
        self.assertEqual(first, b"a1")

        second = await asyncio.wait_for(audio_q.get(), timeout=2)
        self.assertEqual(second, b"a2")
        self.assertTrue(client1.closed)
        self.assertEqual(connect.call_count, 2)

        await text_q.put(SHUTDOWN)
        await asyncio.wait_for(worker, timeout=2)
        self.assertTrue(client2.closed)

    async def test_connect_failure_falls_back_to_secondary_provider(self) -> None:
        from unittest.mock import patch

        from tts_streamer import _connect_provider, TTSConfig

        class _AlwaysFails:
            def __init__(self) -> None:
                self.sample_rate = 24000
                self.num_channels = 1

            async def connect(self) -> None:
                raise ConnectionRefusedError("primary provider unreachable")

        good_client = _FakeProviderClient([])

        calls = {"fish": 0, "sarvam": 0}

        def fake_build(provider: str, config: TTSConfig, *, api_key_override=None):
            calls[provider] += 1
            if provider == "fish":
                return _AlwaysFails()
            return good_client

        import tts_streamer as mod

        original = mod.build_provider_client
        mod.build_provider_client = fake_build

        async def _no_sleep(_delay: float) -> None:
            return

        try:
            config = TTSConfig(
                provider="fish",
                api_key="k",
                voice="v",
                language="en-IN",
                sample_rate=24000,
                fallback_provider="sarvam",
                fallback_api_key="k2",
            )
            # Skip real backoff delays — this test only cares that a
            # connect failure on the primary falls through to the
            # fallback provider, not the retry timing itself.
            with patch("tts_streamer.asyncio.sleep", _no_sleep):
                client = await _connect_provider(config)
        finally:
            mod.build_provider_client = original

        self.assertIs(client, good_client)
        self.assertGreaterEqual(calls["fish"], 1)
        self.assertEqual(calls["sarvam"], 1)


if __name__ == "__main__":
    unittest.main()
