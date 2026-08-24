"""
Unit tests for the Dual-Stage Turn-Taking Arbiter, using an in-memory fake
AsyncOpenAI-shaped client — no network calls.

Run with:
    python -m unittest test_turn_arbiter -v
"""

from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace

from turn_arbiter import (
    RollingTranscriptBuffer,
    TurnArbiter,
    TurnClassification,
    classify_turn_intent,
)


def _completion(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class _FakeCompletions:
    def __init__(self, content: str | None = None, *, delay: float = 0.0, raise_exc: Exception | None = None):
        self._content = content
        self._delay = delay
        self._raise_exc = raise_exc
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._raise_exc:
            raise self._raise_exc
        return _completion(self._content)


class _FakeClient:
    def __init__(self, content: str | None = None, *, delay: float = 0.0, raise_exc: Exception | None = None):
        self.chat = SimpleNamespace(completions=_FakeCompletions(content, delay=delay, raise_exc=raise_exc))


class RollingTranscriptBufferTests(unittest.TestCase):
    def test_window_text_is_latest_event(self) -> None:
        buf = RollingTranscriptBuffer(window_seconds=4.0)
        buf.add("I went", is_final=False, confidence=0.9, timestamp=0.0)
        buf.add("I went to the", is_final=False, confidence=0.8, timestamp=0.5)
        buf.add("I went to the store.", is_final=True, confidence=0.95, timestamp=1.0)
        self.assertEqual(buf.window_text(), "I went to the store.")

    def test_evicts_events_outside_window(self) -> None:
        buf = RollingTranscriptBuffer(window_seconds=2.0)
        buf.add("old", is_final=True, confidence=1.0, timestamp=0.0)
        buf.add("new", is_final=True, confidence=1.0, timestamp=5.0)  # 5s later, outside 2s window
        events = buf.events()
        self.assertEqual([e.text for e in events], ["new"])

    def test_average_confidence(self) -> None:
        buf = RollingTranscriptBuffer(window_seconds=10.0)
        buf.add("a", is_final=False, confidence=0.8, timestamp=0.0)
        buf.add("b", is_final=True, confidence=1.0, timestamp=0.1)
        self.assertAlmostEqual(buf.average_confidence(), 0.9)

    def test_time_since_last_update(self) -> None:
        buf = RollingTranscriptBuffer(window_seconds=10.0)
        buf.add("hello", is_final=True, confidence=1.0, timestamp=10.0)
        self.assertAlmostEqual(buf.time_since_last_update(now=10.3), 0.3, places=5)

    def test_empty_buffer(self) -> None:
        buf = RollingTranscriptBuffer()
        self.assertEqual(buf.window_text(), "")
        self.assertIsNone(buf.latest())
        self.assertIsNone(buf.average_confidence())
        self.assertIsNone(buf.time_since_last_update())

    def test_terminal_punctuation_tracked_per_event(self) -> None:
        buf = RollingTranscriptBuffer()
        buf.add("wait, what", is_final=False, confidence=0.9)
        buf.add("wait, what?", is_final=True, confidence=0.9)
        self.assertFalse(buf.events()[0].ends_with_terminal_punct)
        self.assertTrue(buf.events()[1].ends_with_terminal_punct)


class FastPathTests(unittest.TestCase):
    def test_backchannels_detected(self) -> None:
        from turn_arbiter import _fast_path_classify

        for word in ["yeah", "Yeah.", "uh-huh", "okay", "mm-hmm", "gotcha", "right?"]:
            result = _fast_path_classify(word)
            self.assertIsNotNone(result, f"{word!r} should fast-path")
            self.assertTrue(result.is_backchannel, f"{word!r} should be a backchannel")
            self.assertFalse(result.is_complete_turn)

    def test_dangling_connectors_detected(self) -> None:
        from turn_arbiter import _fast_path_classify

        for text in ["and", "because", "so I was thinking and", "my"]:
            result = _fast_path_classify(text)
            self.assertIsNotNone(result, f"{text!r} should fast-path")
            self.assertTrue(result.requires_pause_extension, f"{text!r} should need extension")
            self.assertFalse(result.is_complete_turn)

    def test_complete_sentence_detected(self) -> None:
        from turn_arbiter import _fast_path_classify

        result = _fast_path_classify("I went to the store yesterday.")
        self.assertIsNotNone(result)
        self.assertTrue(result.is_complete_turn)
        self.assertFalse(result.is_backchannel)
        self.assertFalse(result.requires_pause_extension)

    def test_ambiguous_transcript_escalates(self) -> None:
        from turn_arbiter import _fast_path_classify

        # No terminal punctuation, no dangling connector, not a backchannel —
        # genuinely needs semantic judgment.
        self.assertIsNone(_fast_path_classify("I think"))

        # Empty text is a defined fast-path case (not ambiguous) — always
        # an incomplete, non-backchannel turn.
        empty_result = _fast_path_classify("")
        self.assertIsNotNone(empty_result)
        self.assertFalse(empty_result.is_complete_turn)


class ClassifyTurnIntentTests(unittest.IsolatedAsyncioTestCase):
    async def test_fast_path_bypasses_llm_entirely(self) -> None:
        client = _FakeClient(content="should never be read")
        result = await classify_turn_intent("yeah", client=client)
        self.assertTrue(result.is_backchannel)
        self.assertEqual(client.chat.completions.calls, [])  # LLM never called

    async def test_ambiguous_transcript_calls_llm_and_parses_response(self) -> None:
        payload = json.dumps(
            {"is_complete_turn": True, "is_backchannel": False, "requires_pause_extension": False}
        )
        client = _FakeClient(content=payload)
        result = await classify_turn_intent("I think", client=client, model="test-model")

        self.assertEqual(result, TurnClassification(**json.loads(payload)))
        self.assertEqual(len(client.chat.completions.calls), 1)
        call = client.chat.completions.calls[0]
        self.assertEqual(call["model"], "test-model")
        self.assertEqual(call["messages"][1]["content"], "I think")
        self.assertEqual(call["response_format"], {"type": "json_object"})

    async def test_timeout_falls_back_without_raising(self) -> None:
        client = _FakeClient(content="{}", delay=1.0)  # slower than the timeout
        result = await classify_turn_intent("I think", client=client, timeout=0.01)
        # falls back to the deterministic heuristic — "I think" has no
        # terminal punctuation, so the fallback treats it as incomplete.
        self.assertFalse(result.is_complete_turn)
        self.assertFalse(result.is_backchannel)

    async def test_provider_error_falls_back_without_raising(self) -> None:
        client = _FakeClient(raise_exc=ConnectionError("provider down"))
        result = await classify_turn_intent("I think", client=client, timeout=1.0)
        self.assertFalse(result.is_complete_turn)

    async def test_malformed_json_falls_back_without_raising(self) -> None:
        client = _FakeClient(content="not json at all")
        result = await classify_turn_intent("because", client=client, timeout=1.0)
        # fallback heuristic still catches the dangling connector correctly
        self.assertTrue(result.requires_pause_extension)


class TurnArbiterTests(unittest.IsolatedAsyncioTestCase):
    async def test_evaluate_feeds_buffer_and_classifies(self) -> None:
        arbiter = TurnArbiter(window_seconds=4.0)
        result = await arbiter.evaluate("okay", is_final=True, confidence=0.9)
        self.assertTrue(result.is_backchannel)
        self.assertEqual(arbiter.buffer.window_text(), "okay")

    async def test_reset_clears_buffer(self) -> None:
        arbiter = TurnArbiter()
        await arbiter.evaluate("yeah", is_final=True)
        arbiter.reset()
        self.assertEqual(arbiter.buffer.window_text(), "")


if __name__ == "__main__":
    unittest.main()
