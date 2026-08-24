"""
End-to-end simulation of the SemanticTurnCoordinator state machine —
verifies the VAD -> arbiter -> barge-in/commit wiring in agent.py without a
real LiveKit room, STT, or LLM call.

All scenarios below deliberately use transcripts that resolve on
turn_arbiter's regex fast path (backchannel / dangling-connector / obvious
complete sentence), so classify_turn_intent never touches the network — the
one exception (test_ambiguous_transcript_escalates_to_llm) monkeypatches
agent.classify_turn_intent directly to simulate the LLM path without a real
provider call.

Run with:
    python -m unittest test_turn_orchestrator -v
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

import agent


class _FakeSession:
    """Minimal stand-in for the AgentSession methods SemanticTurnCoordinator
    and InterruptController touch."""

    def __init__(self) -> None:
        self.cleared_turns = 0
        self.committed_turns: list[dict] = []
        self.interrupt_calls = 0

    def clear_user_turn(self) -> None:
        self.cleared_turns += 1

    async def interrupt(self, *, force: bool = False) -> None:
        self.interrupt_calls += 1

    def commit_user_turn(self, **kwargs) -> "asyncio.Future[str]":
        self.committed_turns.append(kwargs)
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        fut.set_result("")
        return fut


async def _settle() -> None:
    """Let scheduled callbacks/tasks run without a fixed sleep duration."""
    for _ in range(20):
        await asyncio.sleep(0)


class SemanticTurnCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.session = _FakeSession()
        self.interrupt_ctrl = agent.InterruptController(self.session)
        self.coordinator = agent.SemanticTurnCoordinator(self.session, self.interrupt_ctrl)

    async def asyncTearDown(self) -> None:
        await self.coordinator.aclose()

    # -- Scenario 1: backchannel --------------------------------------------

    async def test_backchannel_does_not_interrupt_or_commit(self) -> None:
        self.coordinator.on_transcript("yeah", is_final=True)
        self.coordinator.on_speech_ended()
        await self.coordinator._pending_task

        self.assertEqual(self.session.cleared_turns, 1)
        self.assertEqual(self.session.committed_turns, [])
        self.assertFalse(self.interrupt_ctrl.interrupt_event.is_set())
        # bot audio was never touched — nothing to assert beyond "no
        # interruption fired", which the flag above already covers.

    # -- Scenario 2: complete turn -> barge-in kill switch + commit --------

    async def test_complete_turn_fires_barge_in_and_commits(self) -> None:
        self.coordinator.on_transcript("I went to the store yesterday.", is_final=True)
        self.coordinator.on_speech_ended()
        await self.coordinator._pending_task

        self.assertTrue(self.interrupt_ctrl.interrupt_event.is_set())
        self.assertEqual(self.session.interrupt_calls, 1)
        self.assertEqual(len(self.session.committed_turns), 1)
        self.assertEqual(self.session.cleared_turns, 0)

    # -- Scenario 3: mid-thought pause -> extend then commit ---------------

    async def test_pause_extension_commits_after_delay_if_silence_continues(self) -> None:
        original_delta = agent.PAUSE_EXTENSION_DELTA_S
        agent.PAUSE_EXTENSION_DELTA_S = 0.02  # keep the test fast
        try:
            self.coordinator.on_transcript("so I was thinking and", is_final=False)
            self.coordinator.on_speech_ended()

            # immediately after: still waiting out the extension, nothing committed yet
            await asyncio.sleep(0)
            self.assertEqual(self.session.committed_turns, [])

            await self.coordinator._pending_task  # wait for the extension to elapse

            self.assertEqual(len(self.session.committed_turns), 1)
            self.assertTrue(self.interrupt_ctrl.interrupt_event.is_set())
        finally:
            agent.PAUSE_EXTENSION_DELTA_S = original_delta

    # -- Scenario 4: new speech cancels a pending decision cleanly ---------

    async def test_new_speech_cancels_pending_extension_without_dangling_task(self) -> None:
        original_delta = agent.PAUSE_EXTENSION_DELTA_S
        agent.PAUSE_EXTENSION_DELTA_S = 1.0  # long enough to reliably interrupt
        try:
            self.coordinator.on_transcript("so I was thinking and", is_final=False)
            self.coordinator.on_speech_ended()
            await asyncio.sleep(0)  # let the pending task start waiting

            pending = self.coordinator._pending_task
            self.assertIsNotNone(pending)
            self.assertFalse(pending.done())

            # user resumed speaking before the extension elapsed
            self.coordinator.on_speech_started()
            await _settle()

            self.assertTrue(pending.done())
            self.assertTrue(pending.cancelled())
            self.assertEqual(self.session.committed_turns, [])
            self.assertFalse(self.interrupt_ctrl.interrupt_event.is_set())
        finally:
            agent.PAUSE_EXTENSION_DELTA_S = original_delta

    # -- Scenario 5: ambiguous transcript escalates to the LLM path --------

    async def test_ambiguous_transcript_escalates_to_llm(self) -> None:
        from turn_arbiter import TurnClassification

        async def fake_classify_intent(transcript: str):
            self.assertEqual(transcript, "I think")
            return TurnClassification(
                is_complete_turn=True, is_backchannel=False, requires_pause_extension=False
            )

        with patch("agent.classify_turn_intent", fake_classify_intent):
            self.coordinator.on_transcript("I think", is_final=True)
            self.coordinator.on_speech_ended()
            await self.coordinator._pending_task

        self.assertEqual(len(self.session.committed_turns), 1)
        self.assertTrue(self.interrupt_ctrl.interrupt_event.is_set())

    # -- Cleanliness: aclose() cancels a pending task without raising ------

    async def test_aclose_cancels_pending_task_cleanly(self) -> None:
        original_delta = agent.PAUSE_EXTENSION_DELTA_S
        agent.PAUSE_EXTENSION_DELTA_S = 1.0
        try:
            self.coordinator.on_transcript("and", is_final=False)
            self.coordinator.on_speech_ended()
            await asyncio.sleep(0)
            pending = self.coordinator._pending_task
            self.assertFalse(pending.done())

            await self.coordinator.aclose()  # must not raise

            self.assertTrue(pending.done())
            self.assertTrue(pending.cancelled())
        finally:
            agent.PAUSE_EXTENSION_DELTA_S = original_delta


if __name__ == "__main__":
    unittest.main()
