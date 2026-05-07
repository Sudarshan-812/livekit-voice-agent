import json
import logging
import os
from typing import Annotated

from dotenv import load_dotenv
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

load_dotenv()

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
    return rag.search_knowledge_base(query)


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

    agent = Agent(
        instructions=instructions,
        tools=[search_knowledge_base],
        stt=deepgram.STT(),
        llm=google.LLM(
            model="gemini-2.5-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
        ),
        tts=deepgram.TTS(model=voice),
        vad=silero.VAD.load(),
    )

    session = AgentSession()
    await session.start(agent, room=ctx.room)
    logger.info("session started — voice and text input active")

    await session.say(
        "Hello! I'm ready to help. You can ask me anything, or about your uploaded documents.",
        allow_interruptions=True,
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
