"""
Dual-Stage Turn-Taking Arbiter.

Solves two specific false-positive turn-taking triggers in the voice
pipeline:

  1. "Backchannel interruption" — the user says "yeah" / "uh-huh" / "okay"
     while the agent is talking, and something downstream mistakes it for
     the start of a real turn.
  2. "Mid-thought pause" — the user trails off on a conjunction or
     preposition ("...and", "...because", "my...") and the endpointing
     logic fires before they're actually done.

Stage 1 (RollingTranscriptBuffer + regex fast path) is a deterministic,
sub-millisecond check that catches the common cases without ever touching
the network. Stage 2 (classify_turn_intent's LLM call) only runs when the
fast path can't confidently decide, using the same fast OpenAI-compatible
client from llm_stream.py (Groq/Cerebras), bounded by a hard timeout so a
slow or dead provider can never stall turn-taking — on timeout or error it
falls back to the same heuristics as the regex stage.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel
from openai import AsyncOpenAI

from llm_stream import LLMConfig, build_client

logger = logging.getLogger("nexus.turn_arbiter")

DEFAULT_WINDOW_SECONDS = 4.0
DEFAULT_TIMEOUT_MS = 100


class TurnClassification(BaseModel):
    is_complete_turn: bool
    is_backchannel: bool  # e.g., 'yeah', 'uh-huh', 'okay', 'right'
    requires_pause_extension: bool  # paused on conjunction/preposition ('and', 'because', 'my...')


# ── Stage 1a: rolling transcript buffer ──────────────────────────────────────


_TERMINAL_PUNCT_RE = re.compile(r"[.?!]\s*$")


def _ends_with_terminal_punctuation(text: str) -> bool:
    return bool(_TERMINAL_PUNCT_RE.search(text.strip()))


@dataclass(frozen=True)
class TranscriptEvent:
    text: str
    timestamp: float
    is_final: bool
    confidence: float
    ends_with_terminal_punct: bool = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ends_with_terminal_punct", _ends_with_terminal_punctuation(self.text))


class RollingTranscriptBuffer:
    """Keeps the last `window_seconds` of Deepgram interim/final transcripts.

    Deepgram interims are cumulative re-guesses of the current utterance
    until a final closes it, so each new event's text supersedes the
    previous one for "what did they just say" purposes — the buffer keeps
    the whole rolling history (for confidence/timing analysis) but exposes
    the latest event's text as the current transcript.
    """

    def __init__(self, window_seconds: float = DEFAULT_WINDOW_SECONDS) -> None:
        self._window = window_seconds
        self._events: "deque[TranscriptEvent]" = deque()

    def add(
        self,
        text: str,
        *,
        is_final: bool,
        confidence: float = 1.0,
        timestamp: Optional[float] = None,
    ) -> TranscriptEvent:
        ts = timestamp if timestamp is not None else time.monotonic()
        event = TranscriptEvent(text=text, timestamp=ts, is_final=is_final, confidence=confidence)
        self._events.append(event)
        self._evict_stale(ts)
        return event

    def _evict_stale(self, now: float) -> None:
        cutoff = now - self._window
        while self._events and self._events[0].timestamp < cutoff:
            self._events.popleft()

    def events(self) -> list[TranscriptEvent]:
        # Eviction happens on add(), relative to that event's own timestamp
        # — not here — so callers that pass explicit timestamps (tests, or
        # a non-monotonic wall clock) get consistent behavior regardless of
        # how much real time has passed since the last add().
        return list(self._events)

    def latest(self) -> Optional[TranscriptEvent]:
        return self._events[-1] if self._events else None

    def window_text(self) -> str:
        latest = self.latest()
        return latest.text if latest else ""

    def time_since_last_update(self, now: Optional[float] = None) -> Optional[float]:
        latest = self.latest()
        if latest is None:
            return None
        now = now if now is not None else time.monotonic()
        return now - latest.timestamp

    def average_confidence(self) -> Optional[float]:
        evs = self.events()
        if not evs:
            return None
        return sum(e.confidence for e in evs) / len(evs)

    def clear(self) -> None:
        self._events.clear()


# ── Stage 1b: regex fast path ────────────────────────────────────────────────

_BACKCHANNEL_WORDS = [
    "yeah", "yeahh", "yep", "yup", "yea", "uh-huh", "uh huh", "mhm", "mm-hmm",
    "mmhmm", "mm", "okay", "ok", "k", "right", "sure", "got it", "gotcha",
    "i see", "true", "totally", "for real", "no way", "wow", "hm", "hmm",
    "cool", "nice", "alright", "all right",
]
# Longest-first so e.g. "uh huh" matches before a bare "uh" could.
_BACKCHANNEL_RE = re.compile(
    r"^\s*(?:"
    + "|".join(re.escape(w) for w in sorted(_BACKCHANNEL_WORDS, key=len, reverse=True))
    + r")[\s.,!?]*$",
    re.IGNORECASE,
)

_DANGLING_CONNECTORS = {
    "and", "but", "so", "because", "or", "if", "when", "while", "since",
    "although", "though", "unless", "until", "my", "your", "his", "her",
    "our", "their", "the", "a", "an", "to", "with", "for", "that", "which",
    "of", "in", "on", "at", "is", "was", "um", "uh",
}
_TRAILING_WORD_RE = re.compile(r"([a-zA-Z']+)[\s.,!?]*$")


def _ends_on_dangling_connector(text: str) -> bool:
    if not text or _ends_with_terminal_punctuation(text):
        return False
    match = _TRAILING_WORD_RE.search(text.lower())
    return bool(match and match.group(1) in _DANGLING_CONNECTORS)


def _fast_path_classify(transcript: str) -> Optional["TurnClassification"]:
    """Deterministic, sub-millisecond classification. Returns None when the
    transcript doesn't confidently match a known pattern, so the caller can
    escalate to the LLM."""
    stripped = transcript.strip()
    if not stripped:
        return TurnClassification(
            is_complete_turn=False, is_backchannel=False, requires_pause_extension=False
        )
    if _BACKCHANNEL_RE.match(stripped):
        return TurnClassification(
            is_complete_turn=False, is_backchannel=True, requires_pause_extension=False
        )
    if _ends_on_dangling_connector(stripped):
        return TurnClassification(
            is_complete_turn=False, is_backchannel=False, requires_pause_extension=True
        )
    if _ends_with_terminal_punctuation(stripped) and len(stripped.split()) >= 4:
        # A clearly multi-word, terminally-punctuated sentence — treat as
        # complete rather than spend an LLM round trip confirming the obvious.
        return TurnClassification(
            is_complete_turn=True, is_backchannel=False, requires_pause_extension=False
        )
    return None


def _fallback_classify(transcript: str) -> "TurnClassification":
    """Used when the LLM call errors or times out — same regex heuristics
    as the fast path, but always returns a decision instead of None."""
    stripped = transcript.strip()
    if _BACKCHANNEL_RE.match(stripped):
        return TurnClassification(
            is_complete_turn=False, is_backchannel=True, requires_pause_extension=False
        )
    if _ends_on_dangling_connector(stripped):
        return TurnClassification(
            is_complete_turn=False, is_backchannel=False, requires_pause_extension=True
        )
    return TurnClassification(
        is_complete_turn=_ends_with_terminal_punctuation(stripped),
        is_backchannel=False,
        requires_pause_extension=False,
    )


# ── Stage 2: structured-output LLM classification ───────────────────────────


_CLASSIFIER_SYSTEM_PROMPT = (
    "You are a real-time turn-taking classifier for a voice assistant. You are only "
    "called when a fast pattern match couldn't confidently decide, so expect genuinely "
    "ambiguous input — judge intent from the words alone, not tone (you have no audio).\n\n"
    "Given the user's most recent speech transcript (which may be an ASR interim, "
    "possibly cut off mid-word or missing punctuation), decide three things and reply "
    "with ONLY a compact JSON object matching this exact schema — no prose, no markdown, "
    "no explanation:\n"
    '{"is_complete_turn": bool, "is_backchannel": bool, "requires_pause_extension": bool}\n\n'
    "- is_complete_turn: true if the user has finished their thought and it's the "
    "assistant's turn to respond. A short but self-contained answer counts as complete "
    "(e.g. 'the blue one', 'no thanks') — completeness is about whether the thought is "
    "finished, not about length.\n"
    "- is_backchannel: true only if the entire transcript is a short acknowledgement with "
    "no new information — 'yeah', 'uh-huh', 'okay', 'right', 'got it' — said in passing, "
    "not as a direct answer to a question.\n"
    "- requires_pause_extension: true if the transcript trails off on a conjunction, "
    "preposition, or unfinished clause/possessive ('and', 'because', 'so I was...', "
    "'my...'), suggesting the user paused mid-thought and is likely to keep talking.\n\n"
    "At most one of the three should be true. If none clearly apply, default to "
    "is_complete_turn — never leave all three false without a specific reason to.\n\n"
    "Examples:\n"
    '"yeah" -> {"is_complete_turn": false, "is_backchannel": true, "requires_pause_extension": false}\n'
    '"yeah, the second one" -> {"is_complete_turn": true, "is_backchannel": false, "requires_pause_extension": false}\n'
    '"so I wanted to ask about" -> {"is_complete_turn": false, "is_backchannel": false, "requires_pause_extension": true}\n'
    '"can you check my calendar for" -> {"is_complete_turn": false, "is_backchannel": false, "requires_pause_extension": true}\n'
    '"what time is it" -> {"is_complete_turn": true, "is_backchannel": false, "requires_pause_extension": false}\n'
    '"no I don\'t think so" -> {"is_complete_turn": true, "is_backchannel": false, "requires_pause_extension": false}'
)

# Lazily-created, process-wide client — reused across calls so classification
# never pays a fresh TCP/TLS handshake on top of the LLM round trip.
_default_client: Optional[AsyncOpenAI] = None
_default_client_lock = asyncio.Lock()


async def _get_default_client() -> AsyncOpenAI:
    global _default_client
    if _default_client is None:
        async with _default_client_lock:
            if _default_client is None:
                _default_client = build_client()
    return _default_client


async def classify_turn_intent(
    transcript: str,
    *,
    client: Optional[AsyncOpenAI] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    use_fast_path: bool = True,
) -> TurnClassification:
    """Classify whether `transcript` is a complete turn, a backchannel, or a
    mid-thought pause needing more time.

    Checks the regex fast path first (task 3's "bypass the LLM call
    entirely" requirement); only escalates to a structured-output LLM call
    for genuinely ambiguous transcripts. The LLM call is bounded by
    `timeout` (default TURN_ARBITER_TIMEOUT_MS, 100ms) and never raises —
    on timeout, provider error, or a malformed response it falls back to
    the same heuristics as the fast path, so a slow/dead provider can never
    stall turn-taking.
    """
    if use_fast_path:
        fast = _fast_path_classify(transcript)
        if fast is not None:
            return fast

    resolved_model = model or os.getenv("TURN_ARBITER_MODEL") or LLMConfig.from_env().model
    resolved_timeout = (
        timeout if timeout is not None else float(os.getenv("TURN_ARBITER_TIMEOUT_MS", str(DEFAULT_TIMEOUT_MS))) / 1000
    )
    resolved_client = client or await _get_default_client()

    try:
        response = await asyncio.wait_for(
            resolved_client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": transcript},
                ],
                temperature=0,
                max_tokens=60,
                response_format={"type": "json_object"},
            ),
            timeout=resolved_timeout,
        )
        content = response.choices[0].message.content or "{}"
        return TurnClassification.model_validate_json(content)
    except Exception as exc:  # noqa: BLE001 — provider errors, timeouts, bad JSON all fall back
        logger.warning("[turn-arbiter] LLM classification failed (%s) — using fallback heuristic", exc)
        return _fallback_classify(transcript)


# ── Orchestrator ──────────────────────────────────────────────────────────────


class TurnArbiter:
    """Ties the rolling buffer and classifier together: feed it every STT
    update, get back the current turn classification for the live window."""

    def __init__(
        self,
        *,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.buffer = RollingTranscriptBuffer(window_seconds=window_seconds)
        self._model = model
        self._timeout = timeout

    async def evaluate(
        self,
        text: str,
        *,
        is_final: bool,
        confidence: float = 1.0,
        timestamp: Optional[float] = None,
    ) -> TurnClassification:
        self.buffer.add(text, is_final=is_final, confidence=confidence, timestamp=timestamp)
        return await classify_turn_intent(
            self.buffer.window_text(), model=self._model, timeout=self._timeout
        )

    def reset(self) -> None:
        self.buffer.clear()
