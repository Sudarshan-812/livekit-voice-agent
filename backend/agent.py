import asyncio
import json
import logging
import os
from typing import Annotated
from livekit.plugins import deepgram, groq, silero

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
from livekit.plugins import deepgram, google, silero

import rag

# Explicitly search up the directory tree to find .env (handles running from backend/)
load_dotenv(find_dotenv())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-agent")

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Keep responses concise since they will be spoken aloud. "
    "When the user asks about any uploaded documents or their contents, always use the "
    "search_knowledge_base tool to find accurate information before answering. "
    "Never guess or fabricate document contents — rely solely on the tool."
)


@function_tool(
    description="Searches the uploaded documents for specific information to answer user questions accurately."
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
    # ChromaDB embedding is synchronous and CPU-bound — run in a thread to avoid
    # blocking the event loop (which would freeze audio I/O and LiveKit heartbeats)
    return await asyncio.to_thread(rag.search_knowledge_base, query)


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()
    logger.info("participant joined: %s", participant.identity)

    try:
        meta = json.loads(participant.metadata or "{}")
    except Exception:
        meta = {}

    instructions = meta.get("system_prompt", "").strip() or SYSTEM_PROMPT
    voice = meta.get("voice", "aura-2-andromeda-en")
    logger.info("voice model: %s", voice)

    # Models belong on AgentSession — this is the canonical pipeline owner.
    # Agent holds only the persona (instructions + tools).
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

    agent = Agent(
        instructions=instructions,
        tools=[search_knowledge_base],
    )

    @session.on("user_input_transcribed")
    def on_user_input(ev):
        logger.info("user said [final=%s]: %s", ev.is_final, ev.transcript)

    @session.on("agent_state_changed")
    def on_state(ev):
        logger.info("agent state: %s → %s", ev.old_state, ev.new_state)

    @session.on("error")
    def on_error(ev):
        logger.error("session error: %s", ev.error)

    await session.start(agent, room=ctx.room)
    logger.info("session started — voice and text input active")

    # say() returns a SpeechHandle — no await so the entrypoint returns immediately
    # and the session loop stays free to process user input in parallel.
    session.say(
        "Hello! I'm ready to help. You can ask me anything, or about your uploaded documents.",
        allow_interruptions=True,
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
