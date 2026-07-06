import asyncio
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
from livekit.plugins import deepgram, groq, silero

import rag

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
    VAD fires on_speech_started()
        → sets interrupt_event (asyncio.Event — zero-copy, sub-millisecond signal)
        → _interrupt_watcher() wakes, cancels registered pipeline tasks
        → calls session.interrupt() to flush the LiveKit outbound audio buffer

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

    def on_speech_started(self) -> None:
        """Call this the exact millisecond VAD detects voice onset."""
        if not self.interrupt_event.is_set():
            logger.info("[barge-in] VAD onset → firing interrupt_event")
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
    voice = meta.get("voice", "aura-2-andromeda-en")
    logger.info("voice model: %s", voice)

    session = AgentSession(
        stt=deepgram.STT(),
        llm=groq.LLM(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
        ),
        tts=deepgram.TTS(model=voice),
        vad=silero.VAD.load(
            activation_threshold=0.35,
            min_silence_duration=0.4,
        ),
    )

    # Instantiate controllers before wiring events — prevents a race where
    # the first speech event fires before the handlers are registered.
    interrupt_ctrl = InterruptController(session)
    watchdog = IdleWatchdog(ctx)

    # ── Event wiring ──────────────────────────────────────────────────────────

    @session.on("user_speech_started")
    def on_user_speech_started() -> None:
        # VAD speech onset — triggers barge-in and resets idle timer.
        # This is the "exact millisecond" trigger described in the interrupt design.
        interrupt_ctrl.on_speech_started()
        watchdog.pulse("user_speech_started")

    @session.on("user_speech_ended")
    def on_user_speech_ended() -> None:
        watchdog.pulse("user_speech_ended")

    @session.on("agent_speech_started")
    def on_agent_speech_started() -> None:
        watchdog.pulse("agent_speech_started")

    @session.on("agent_speech_ended")
    def on_agent_speech_ended() -> None:
        watchdog.pulse("agent_speech_ended")

    @session.on("user_input_transcribed")
    def on_user_input(ev) -> None:
        logger.info("user said [final=%s]: %s", ev.is_final, ev.transcript)

    @session.on("agent_state_changed")
    def on_state(ev) -> None:
        logger.info("agent state: %s → %s", ev.old_state, ev.new_state)

    @session.on("error")
    def on_error(ev) -> None:
        logger.error("session error: %s", ev.error)

    # ── Lifecycle: arm background tasks then start session ────────────────────

    interrupt_ctrl.start()
    watchdog.start()

    agent = Agent(
        instructions=instructions,
        tools=[search_knowledge_base],
    )

    await session.start(agent, room=ctx.room)
    logger.info(
        "session started — barge-in controller and idle watchdog (%.0fs) active",
        IDLE_TIMEOUT_SECONDS,
    )

    session.say(
        "Hello! I'm Nexus, ready to help. You can ask me anything, "
        "or about your uploaded documents.",
        allow_interruptions=True,
    )

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
        await interrupt_ctrl.stop()
        await watchdog.stop()
        # FinOps: record final session cost (replace 0s with real counters)
        await track_cost(session_id, prompt_tokens=0, audio_seconds=0.0)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
