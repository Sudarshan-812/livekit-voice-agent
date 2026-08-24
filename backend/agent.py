import asyncio
import contextlib
import json
import logging
import os
import time
from typing import Annotated

from dotenv import find_dotenv, load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import deepgram, silero

import rag
from livekit_audio_publisher import publish_audio_track
from llm_provider import build_llm
from tts_streamer import SHUTDOWN, TURN_END, TTSConfig, stream_tts_worker
from turn_arbiter import RollingTranscriptBuffer, classify_turn_intent

load_dotenv(find_dotenv())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nexus.agent")

IDLE_TIMEOUT_SECONDS: float = float(os.getenv("NEXUS_IDLE_TIMEOUT_SECONDS", "45"))

SYSTEM_PROMPT = (
    "You are Nexus, a helpful voice assistant. Keep responses concise since they will be "
    "spoken aloud. When the user asks about any uploaded documents or their contents, always "
    "use the search_knowledge_base tool to find accurate information before answering. "
    "Never guess or fabricate document contents — rely solely on the tool."
)


# ── FinOps: Redis cost tracking ───────────────────────────────────────────────

async def track_cost(
    session_id: str,
    prompt_tokens: int,
    audio_seconds: float,
) -> None:
    """
    Increment per-session cost counters in Redis.

    Replace the body with aioredis calls, e.g.:
        await redis.incrby(f"nexus:tokens:prompt:{session_id}", prompt_tokens)
        await redis.incrbyfloat(f"nexus:audio:secs:{session_id}", audio_seconds)
        await redis.expire(f"nexus:tokens:prompt:{session_id}", 86400)

    See FINOPS.md §3 for the full key schema.
    """
    logger.debug(
        "[finops] session=%s  prompt_tokens=%d  audio_sec=%.2f",
        session_id,
        prompt_tokens,
        audio_seconds,
    )


# ── Task 2: Barge-in Interrupt Controller ─────────────────────────────────────

class InterruptController:
    """
    Full-duplex asyncio interrupt multiplexer for barge-in support.

    Architecture
    ────────────
    SemanticTurnCoordinator decides the user's speech was a real, complete
    turn (not a backchannel, not a mid-thought pause) and calls trigger()
        → sets interrupt_event (asyncio.Event — zero-copy, sub-millisecond signal)
        → _interrupt_watcher() wakes, cancels registered pipeline tasks
        → calls session.interrupt() to flush the LiveKit outbound audio buffer

    interrupt_event is also watched directly by tts_streamer.stream_tts_worker
    and livekit_audio_publisher.publish_audio_track, so the same trigger()
    call purges the TTS text/audio queues and clears the WebRTC audio
    source's playout buffer — no separate wiring needed for those.

    Registered tasks are any asyncio.Tasks we spawn outside AgentSession
    (e.g. custom LLM streaming, TTS chunk prefetch). The session's own LLM/TTS
    tasks are handled by session.interrupt() on the LiveKit side.

    CancelledError is caught inside each registered task's own except block;
    _interrupt_watcher does not need to await them after cancellation.
    """

    def __init__(self, session: AgentSession) -> None:
        self._session = session
        self.interrupt_event: asyncio.Event = asyncio.Event()
        self._active_tasks: set[asyncio.Task] = set()
        self._watcher_task: asyncio.Task | None = None

    def start(self) -> None:
        self._watcher_task = asyncio.ensure_future(self._interrupt_watcher())

    def register_task(self, task: asyncio.Task) -> asyncio.Task:
        """Track a pipeline task so it is cancelled on the next barge-in."""
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)
        return task

    def trigger(self) -> None:
        """Fire the barge-in kill switch — call this the moment the turn
        arbiter decides the user has completed a real turn."""
        if not self.interrupt_event.is_set():
            logger.info("[barge-in] complete turn confirmed → firing interrupt_event")
            self.interrupt_event.set()

    async def _interrupt_watcher(self) -> None:
        while True:
            try:
                await self.interrupt_event.wait()
                self.interrupt_event.clear()

                # Cancel every active custom pipeline task
                cancelled_count = 0
                for task in list(self._active_tasks):
                    if not task.done():
                        task.cancel()
                        cancelled_count += 1

                if cancelled_count:
                    logger.info(
                        "[barge-in] cancelled %d active pipeline task(s)",
                        cancelled_count,
                    )

                # Flush the LiveKit outbound audio buffer so the speaker stops
                # immediately — session.interrupt() is synchronous in AgentSession
                try:
                    self._session.interrupt()
                    logger.info("[barge-in] LiveKit audio buffer flushed")
                except Exception as exc:
                    logger.warning("[barge-in] buffer flush error: %s", exc)

            except asyncio.CancelledError:
                break

    async def stop(self) -> None:
        if self._watcher_task and not self._watcher_task.done():
            self._watcher_task.cancel()
            try:
                await self._watcher_task
            except asyncio.CancelledError:
                pass


# ── Semantic Turn-Taking Arbiter ──────────────────────────────────────────────

BASE_SILENCE_TIMEOUT_S = 0.4
"""Matches the VAD's own min_silence_duration — by the time "listening" fires,
this much silence has already elapsed, so it's the effective baseline."""

EXTENDED_SILENCE_TIMEOUT_S = 1.2
"""Total silence budget when the arbiter detects a mid-thought pause."""

PAUSE_EXTENSION_DELTA_S = EXTENDED_SILENCE_TIMEOUT_S - BASE_SILENCE_TIMEOUT_S
"""Additional wait beyond the VAD's own 400ms — VAD already spent
BASE_SILENCE_TIMEOUT_S getting here, so only the delta needs waiting out."""

TRANSCRIPT_WINDOW_SECONDS = 5.0


class SemanticTurnCoordinator:
    """
    Dual-stage turn-taking arbiter sitting between Silero VAD and the
    barge-in kill switch — solves "mid-thought pause" and "backchannel
    interruption" by not committing a user turn the instant VAD reports
    silence.

    Wiring (see entrypoint):
      - on_transcript(): fed from every `user_input_transcribed` event,
        keeps RollingTranscriptBuffer current.
      - on_speech_started(): fed from `user_state_changed` -> "speaking"
        (VAD onset). Cancels any pending silence decision — the user kept
        talking, so there's nothing to commit yet.
      - on_speech_ended(): fed from `user_state_changed` -> "listening"
        (VAD silence, already BASE_SILENCE_TIMEOUT_S/400ms in per Silero's
        min_silence_duration). Classifies the buffered transcript and:
          * is_backchannel            -> discard; bot audio keeps playing.
          * requires_pause_extension  -> wait PAUSE_EXTENSION_DELTA_S more
                                         (400ms -> 1200ms total) before
                                         committing, cancellable by new speech.
          * is_complete_turn          -> fire the barge-in kill switch and
                                         commit the turn to the LLM worker.

    Every decision runs in a single cancellable background task, so a new
    VAD onset (on_speech_started) or session teardown (aclose) always wins
    cleanly over an in-flight classification or pause-extension wait —
    asyncio.CancelledError propagates straight out of _decide() with no
    dangling task or unclosed socket left behind (the LLM call inside
    classify_turn_intent is cancelled the same way).
    """

    def __init__(self, session: AgentSession, interrupt_ctrl: InterruptController) -> None:
        self._session = session
        self._interrupt_ctrl = interrupt_ctrl
        self.buffer = RollingTranscriptBuffer(window_seconds=TRANSCRIPT_WINDOW_SECONDS)
        self._pending_task: asyncio.Task | None = None

    def on_transcript(self, transcript: str, *, is_final: bool) -> None:
        self.buffer.add(transcript, is_final=is_final)

    def on_speech_started(self) -> None:
        self._cancel_pending()

    def on_speech_ended(self) -> None:
        self._cancel_pending()
        self._pending_task = asyncio.ensure_future(self._decide())

    def _cancel_pending(self) -> None:
        if self._pending_task and not self._pending_task.done():
            self._pending_task.cancel()

    async def _decide(self) -> None:
        try:
            transcript = self.buffer.window_text()
            result = await classify_turn_intent(transcript)
            logger.info(
                "[turn-arbiter] transcript=%r complete=%s backchannel=%s extend=%s",
                transcript,
                result.is_complete_turn,
                result.is_backchannel,
                result.requires_pause_extension,
            )

            if result.is_backchannel:
                logger.info("[turn-arbiter] backchannel — discarding, bot audio continues")
                self._session.clear_user_turn()
                return

            if result.requires_pause_extension:
                logger.info(
                    "[turn-arbiter] mid-thought pause — extending silence timeout %.0fms -> %.0fms",
                    BASE_SILENCE_TIMEOUT_S * 1000,
                    EXTENDED_SILENCE_TIMEOUT_S * 1000,
                )
                await asyncio.sleep(PAUSE_EXTENSION_DELTA_S)
                logger.info("[turn-arbiter] extension elapsed with no further speech — committing")

            await self._commit_turn()
        except asyncio.CancelledError:
            logger.debug("[turn-arbiter] decision cancelled — user resumed speaking")
            raise

    async def _commit_turn(self) -> None:
        # Barge-in kill switch: purges tts_streamer's audio/text queues,
        # clears the WebRTC playout buffer, and resets the TTS connection —
        # idempotent/harmless if the bot wasn't actually speaking.
        self._interrupt_ctrl.trigger()
        with contextlib.suppress(Exception):
            await self._session.interrupt()
        # Route the full user query to the LLM worker.
        self._session.commit_user_turn(transcript_timeout=2.0)

    async def aclose(self) -> None:
        self._cancel_pending()
        if self._pending_task:
            with contextlib.suppress(asyncio.CancelledError):
                await self._pending_task


# ── Task 3: FinOps Idle Watchdog ──────────────────────────────────────────────

class IdleWatchdog:
    """
    Disconnects the WebRTC session after IDLE_TIMEOUT_SECONDS of inactivity.

    Architecture
    ────────────
    Uses asyncio.wait_for on an asyncio.Event rather than a polling sleep loop:
    - Zero CPU overhead while active
    - Sub-second reaction time on timeout expiry
    - Each call to pulse() resets the timer by re-arming the Event

    On timeout the room is disconnected cleanly, which:
    1. Terminates the LiveKit WebRTC socket
    2. Triggers the room's "disconnected" event, unblocking the entrypoint
    3. Allows the finally block to run track_cost() and task cleanup
    """

    def __init__(self, ctx: JobContext, timeout: float = IDLE_TIMEOUT_SECONDS) -> None:
        self._ctx = ctx
        self._timeout = timeout
        self._activity_event: asyncio.Event = asyncio.Event()
        self._watchdog_task: asyncio.Task | None = None
        self._last_activity: float = time.monotonic()

    def start(self) -> None:
        self._activity_event.set()  # arm immediately on connect
        self._watchdog_task = asyncio.ensure_future(self._watch())

    def pulse(self, reason: str = "") -> None:
        """Reset the idle timer. Call on any user or agent speech activity."""
        self._last_activity = time.monotonic()
        self._activity_event.set()
        if reason:
            logger.debug("[watchdog] activity: %s", reason)

    async def _watch(self) -> None:
        while True:
            self._activity_event.clear()
            try:
                await asyncio.wait_for(
                    self._activity_event.wait(),
                    timeout=self._timeout,
                )
            except asyncio.TimeoutError:
                idle_secs = time.monotonic() - self._last_activity
                logger.warning(
                    "[watchdog] %.1fs idle — disconnecting to purge API costs",
                    idle_secs,
                )
                try:
                    await self._ctx.room.disconnect()
                except Exception as exc:
                    logger.error("[watchdog] disconnect error: %s", exc)
                break

    async def stop(self) -> None:
        if self._watchdog_task and not self._watchdog_task.done():
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass


# ── RAG Tool ──────────────────────────────────────────────────────────────────

@function_tool(
    description=(
        "Searches the uploaded documents for specific information to answer "
        "user questions accurately."
    )
)
async def search_knowledge_base(
    context: RunContext,
    query: Annotated[str, "The search query to look up in the uploaded documents"],
) -> str:
    logger.info("RAG search: %s", query)
    try:
        room = context.session.room_io.room
        payload = json.dumps({"type": "rag_search", "query": query}).encode()
        await room.local_participant.publish_data(payload, topic="rag-status")
    except Exception:
        pass
    # ChromaDB embedding is synchronous and CPU-bound — offload to thread pool
    # to avoid blocking the asyncio event loop (which would stall audio I/O)
    return await asyncio.to_thread(rag.search_knowledge_base, query)


# ── Custom-TTS Agent ────────────────────────────────────────────────────────

class NexusAgent(Agent):
    """Agent whose speech is synthesized by tts_streamer.py's streaming TTS
    worker instead of AgentSession's built-in tts_node.

    AgentSession invokes tts_node once per turn, which would mean opening a
    fresh provider WebSocket per response — tts_streamer.py is built around
    one persistent connection across the whole session instead. So this taps
    llm_node purely as a forwarder: every content delta is pushed onto
    tts_text_queue as it streams (for clause-level, low-latency synthesis),
    while the original chunks are yielded through completely unchanged, so
    AgentSession's tool-calling, chat history, and turn/state tracking work
    exactly as they would with the default implementation. Audio playback
    for this session is disabled at the AgentSession level (see entrypoint)
    so tts_node is never actually invoked.
    """

    def __init__(self, *, instructions: str, tools: list, tts_text_queue: "asyncio.Queue") -> None:
        super().__init__(instructions=instructions, tools=tools)
        self._tts_text_queue = tts_text_queue

    async def llm_node(self, chat_ctx, tools, model_settings):
        async for chunk in super().llm_node(chat_ctx, tools, model_settings):
            if chunk.delta and chunk.delta.content:
                await self._tts_text_queue.put(chunk.delta.content)
            yield chunk
        await self._tts_text_queue.put(TURN_END)


# ── Entrypoint ────────────────────────────────────────────────────────────────

async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()
    session_id = participant.identity
    logger.info("participant joined: %s", session_id)

    try:
        meta = json.loads(participant.metadata or "{}")
    except Exception:
        meta = {}

    instructions = meta.get("system_prompt", "").strip() or SYSTEM_PROMPT
    tts_config = TTSConfig.from_env()
    logger.info(
        "tts provider: %s (voice=%s, sample_rate=%d)",
        tts_config.provider,
        tts_config.voice,
        tts_config.sample_rate,
    )

    # TTS is not wired through AgentSession's tts= plugin — audio output is
    # produced entirely by our own persistent stream_tts_worker (see below),
    # since it holds one WebSocket open across the whole session rather than
    # reconnecting per turn the way AgentSession's tts_node would.
    # turn_detection="manual" hands end-of-turn control entirely to
    # SemanticTurnCoordinator below — AgentSession's own automatic
    # endpointing/interruption-on-activity is disabled (it would otherwise
    # commit/interrupt on every VAD silence or STT activity, which is
    # exactly the "backchannel interruption" / "mid-thought pause" problem
    # this rewire exists to fix). VAD is still required: it's what drives
    # the raw user_state_changed speaking/listening transitions the
    # coordinator reacts to, and min_silence_duration=0.4 is the 400ms
    # baseline silence timeout it extends to 1200ms when needed.
    session = AgentSession(
        stt=deepgram.STT(),
        llm=build_llm(),
        tts=None,
        vad=silero.VAD.load(
            activation_threshold=0.35,
            min_silence_duration=BASE_SILENCE_TIMEOUT_S,
        ),
        turn_handling={"turn_detection": "manual"},
    )

    # Instantiate controllers before wiring events — prevents a race where
    # the first speech event fires before the handlers are registered.
    interrupt_ctrl = InterruptController(session)
    watchdog = IdleWatchdog(ctx)
    turn_coordinator = SemanticTurnCoordinator(session, interrupt_ctrl)

    # ── Streaming TTS pipeline ──────────────────────────────────────────────
    # text_queue: LLM token deltas in (from NexusAgent.llm_node)
    # audio_out_queue: raw PCM16 bytes out, published to a LiveKit audio track
    # Both share interrupt_ctrl.interrupt_event for barge-in — the same
    # signal SemanticTurnCoordinator fires on a confirmed complete turn, so
    # a real interruption purges audio/text buffers and resets the TTS
    # connection with no extra wiring beyond what InterruptController and
    # the coordinator already do.
    tts_text_queue: asyncio.Queue = asyncio.Queue()
    tts_audio_queue: asyncio.Queue = asyncio.Queue()

    tts_worker_task = asyncio.ensure_future(
        stream_tts_worker(
            tts_text_queue,
            tts_audio_queue,
            interrupt_ctrl.interrupt_event,
            config=tts_config,
        )
    )
    audio_publish_task = await publish_audio_track(
        ctx.room,
        tts_audio_queue,
        interrupt_ctrl.interrupt_event,
        sample_rate=tts_config.sample_rate,
    )

    # ── Event wiring ──────────────────────────────────────────────────────────
    # NOTE: "user_speech_started"/"user_speech_ended"/"agent_speech_started"/
    # "agent_speech_ended" are not real AgentSession event names (the actual
    # EventTypes are "user_state_changed" and "agent_state_changed" with an
    # old_state/new_state pair) — those handlers used to silently never
    # fire. Fixed here as part of wiring in the turn arbiter, since it needs
    # the real speaking/listening transitions to work at all.

    @session.on("user_state_changed")
    def on_user_state(ev) -> None:
        if ev.new_state == "speaking":
            # VAD onset — cancel any pending silence decision, the user
            # kept talking, so there's nothing to commit yet.
            turn_coordinator.on_speech_started()
            watchdog.pulse("user_speech_started")
        elif ev.new_state == "listening":
            # VAD silence (already BASE_SILENCE_TIMEOUT_S in) — do NOT
            # commit immediately, classify first.
            turn_coordinator.on_speech_ended()
            watchdog.pulse("user_speech_ended")

    @session.on("user_input_transcribed")
    def on_user_input(ev) -> None:
        logger.info("user said [final=%s]: %s", ev.is_final, ev.transcript)
        turn_coordinator.on_transcript(ev.transcript, is_final=ev.is_final)

    @session.on("agent_state_changed")
    def on_state(ev) -> None:
        logger.info("agent state: %s → %s", ev.old_state, ev.new_state)
        if ev.new_state == "speaking":
            watchdog.pulse("agent_speech_started")
        elif ev.old_state == "speaking":
            watchdog.pulse("agent_speech_ended")

    @session.on("error")
    def on_error(ev) -> None:
        logger.error("session error: %s", ev.error)

    # ── Lifecycle: arm background tasks then start session ────────────────────

    interrupt_ctrl.start()
    watchdog.start()

    agent = NexusAgent(
        instructions=instructions,
        tools=[search_knowledge_base],
        tts_text_queue=tts_text_queue,
    )

    await session.start(agent, room=ctx.room)
    # tts_node is only ever called when output.audio_enabled is True — since
    # our own publish_audio_track already owns the room's audio track, make
    # sure AgentSession never tries to synthesize (or publish) audio itself.
    session.output.set_audio_enabled(False)
    logger.info(
        "session started — barge-in controller and idle watchdog (%.0fs) active",
        IDLE_TIMEOUT_SECONDS,
    )

    greeting = (
        "Hello! I'm Nexus, ready to help. You can ask me anything, "
        "or about your uploaded documents."
    )
    # session.say() still drives chat history / agent_state bookkeeping (its
    # own audio path is a no-op now that output.audio is disabled) — the
    # actual speech comes from pushing the same text through our TTS worker.
    session.say(greeting, allow_interruptions=True)
    await tts_text_queue.put(greeting)
    await tts_text_queue.put(TURN_END)

    # ── Keep entrypoint alive until the room disconnects ──────────────────────
    # Using a Future gated on the "disconnected" room event rather than a sleep
    # loop — this is the correct asyncio pattern: block without spinning.

    disconnect_future: asyncio.Future = asyncio.get_event_loop().create_future()

    @ctx.room.on("disconnected")
    def on_room_disconnected(*_) -> None:
        if not disconnect_future.done():
            disconnect_future.set_result(None)

    try:
        await disconnect_future
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("session ending — running cleanup for %s", session_id)
        await turn_coordinator.aclose()
        await interrupt_ctrl.stop()
        await watchdog.stop()

        await tts_text_queue.put(SHUTDOWN)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(tts_worker_task, timeout=3)
        await tts_audio_queue.put(SHUTDOWN)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(audio_publish_task, timeout=3)

        # FinOps: record final session cost (replace 0s with real counters)
        await track_cost(session_id, prompt_tokens=0, audio_seconds=0.0)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
