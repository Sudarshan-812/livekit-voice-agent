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
    logger.info("Searching knowledge base: %s", query)
    return rag.search_knowledge_base(query)


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()

    instructions = (participant.metadata or "").strip() or SYSTEM_PROMPT

    agent = Agent(
        instructions=instructions,
        tools=[search_knowledge_base],
        stt=deepgram.STT(),
        llm=google.LLM(
            model="gemini-1.5-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
        ),
        tts=google.TTS(
            api_key=os.getenv("GEMINI_API_KEY"),
        ),
        vad=silero.VAD.load(),
    )

    session = AgentSession()
    await session.start(agent, room=ctx.room)
    await session.say(
        "Hello! I'm ready to help. You can ask me anything, or about your uploaded documents.",
        allow_interruptions=True,
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
