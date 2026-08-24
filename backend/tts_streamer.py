"""
Async streaming TTS worker — connects directly to a provider's WebSocket
streaming API (Fish Audio or Sarvam) rather than a batch/REST TTS call.

    stream_tts_worker(text_queue, audio_out_queue, interruption_event)

consumes LLM tokens from ``text_queue``, buffers them into clauses (flushed
on ',', '.', '?', '!', '\\n'), streams each clause into a persistent
provider WebSocket, and pushes raw PCM16 audio bytes onto
``audio_out_queue`` the instant they arrive — no batching, no waiting for
a full utterance.

Providers (env-driven, see TTSConfig):
    TTS_PROVIDER            fish | sarvam            (default: fish)
    TTS_API_KEY
    TTS_VOICE                                        (reference_id / speaker)
    TTS_LANGUAGE                                      (Sarvam only, default en-IN)
    TTS_SAMPLE_RATE                                    (default: 24000)
    TTS_FALLBACK_PROVIDER   fish | sarvam            (optional)
    TTS_FALLBACK_API_KEY                               (optional, defaults to TTS_API_KEY)

Protocol notes (verified against provider docs — not guessed):
  Fish Audio (wss://api.fish.audio/v1/tts/live): MessagePack-framed events
  {start, text, flush, stop} out / {audio, finish} in. Auth via
  ``Authorization: Bearer <key>`` header.
  https://docs.fish.audio/api-reference/endpoint/websocket/tts-live

  Sarvam (wss://api.sarvam.ai/text-to-speech/ws): JSON events
  {config, text, flush, ping} out / {audio (base64), event, error} in.
  Auth via ``api-subscription-key`` header.
  https://docs.sarvam.ai/api-reference-docs/text-to-speech-streaming/stream

Neither provider exposes a server-side "cancel this utterance but keep the
socket" primitive, so a barge-in reset (task 3) is implemented as: drain
both queues, then close and reopen the WebSocket. That still keeps the
connection persistent across normal turn boundaries (task 2) — a reconnect
only happens on interruption or an actual socket drop (task 4).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Optional

import msgpack
import websockets

logger = logging.getLogger("nexus.tts.stream")

# Pushed onto text_queue to mark "this LLM turn's tokens are complete" —
# flush the trailing partial clause, but keep the worker/connection alive.
TURN_END = object()

# Pushed onto text_queue to stop the worker entirely (session teardown).
SHUTDOWN = object()

CLAUSE_BOUNDARIES = {",", ".", "?", "!", "\n"}


# ── Clause buffering ───────────────────────────────────────────────────────


class ClauseBuffer:
    """Accumulates LLM tokens and yields complete clauses on punctuation.

    Iterates character-by-character since a single LLM token can contain
    zero, one, or several clause boundaries (tokens rarely align with
    punctuation), so this is the only way to catch every boundary.
    """

    def __init__(self) -> None:
        self._buf: list[str] = []

    def push(self, token: str) -> list[str]:
        clauses: list[str] = []
        for ch in token:
            self._buf.append(ch)
            if ch in CLAUSE_BOUNDARIES:
                clause = "".join(self._buf).strip()
                self._buf.clear()
                if clause:
                    clauses.append(clause)
        return clauses

    def drain(self) -> Optional[str]:
        """Flush a trailing partial clause (e.g. end of turn). None if empty."""
        if not self._buf:
            return None
        clause = "".join(self._buf).strip()
        self._buf.clear()
        return clause or None

    def reset(self) -> None:
        self._buf.clear()


# ── Errors ──────────────────────────────────────────────────────────────────


class TTSConnectionError(Exception):
    """Raised when a provider WebSocket cannot be established after retries."""


class TTSProviderError(Exception):
    """Raised when the provider sends an explicit in-band error message."""


# ── Config ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TTSConfig:
    provider: str
    api_key: Optional[str]
    voice: str
    language: str
    sample_rate: int
    fallback_provider: Optional[str]
    fallback_api_key: Optional[str]

    @classmethod
    def from_env(cls) -> "TTSConfig":
        return cls(
            provider=os.getenv("TTS_PROVIDER", "fish").strip().lower(),
            api_key=os.getenv("TTS_API_KEY"),
            voice=os.getenv("TTS_VOICE", "default"),
            language=os.getenv("TTS_LANGUAGE", "en-IN"),
            sample_rate=int(os.getenv("TTS_SAMPLE_RATE", "24000")),
            fallback_provider=(os.getenv("TTS_FALLBACK_PROVIDER") or "").strip().lower() or None,
            fallback_api_key=os.getenv("TTS_FALLBACK_API_KEY"),
        )


# ── Provider client interface ────────────────────────────────────────────────


class TTSProviderClient(ABC):
    """Minimal interface every provider WebSocket client implements."""

    sample_rate: int
    num_channels: int = 1

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def send_text(self, text: str) -> None:
        """Stream one clause of text into the provider's input buffer."""

    @abstractmethod
    async def flush(self) -> None:
        """Force the provider to synthesize whatever text is buffered so far."""

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    def audio_chunks(self) -> AsyncIterator[bytes]:
        """Yield raw audio bytes as they arrive. Ends on close/drop/error."""


# ── Fish Audio client ────────────────────────────────────────────────────────

FISH_DEFAULT_URL = "wss://api.fish.audio/v1/tts/live"


class FishAudioClient(TTSProviderClient):
    def __init__(
        self,
        *,
        api_key: Optional[str],
        voice: str,
        sample_rate: int = 24000,
        model: str = "s1",
        url: str = FISH_DEFAULT_URL,
    ) -> None:
        self.sample_rate = sample_rate
        self.num_channels = 1
        self._api_key = api_key
        self._voice = voice
        self._model = model
        self._url = url
        self._ws: Optional[Any] = None

    async def connect(self) -> None:
        self._ws = await websockets.connect(
            self._url,
            additional_headers={
                "Authorization": f"Bearer {self._api_key}",
                "model": self._model,
            },
        )
        await self._send_event(
            {
                "event": "start",
                "request": {
                    "text": "",
                    "format": "pcm",
                    "chunk_length": 200,
                    "reference_id": self._voice,
                    "latency": "low",
                    "sample_rate": self.sample_rate,
                },
            }
        )

    async def _send_event(self, payload: dict) -> None:
        if self._ws is None:
            raise TTSConnectionError("Fish Audio client used before connect()")
        await self._ws.send(msgpack.packb(payload, use_bin_type=True))

    async def send_text(self, text: str) -> None:
        await self._send_event({"event": "text", "text": text})

    async def flush(self) -> None:
        await self._send_event({"event": "flush"})

    async def close(self) -> None:
        if self._ws is None:
            return
        with contextlib.suppress(Exception):
            await self._send_event({"event": "stop"})
        with contextlib.suppress(Exception):
            await self._ws.close()
        self._ws = None

    async def audio_chunks(self) -> AsyncIterator[bytes]:
        if self._ws is None:
            raise TTSConnectionError("Fish Audio client used before connect()")
        async for raw in self._ws:
            msg = msgpack.unpackb(raw, raw=False)
            event = msg.get("event")
            if event == "audio":
                audio = msg.get("audio")
                if audio:
                    yield audio
            elif event == "finish":
                if msg.get("reason") == "error":
                    raise TTSProviderError("Fish Audio reported a synthesis error")
                return


# ── Sarvam client ─────────────────────────────────────────────────────────────

SARVAM_DEFAULT_URL = "wss://api.sarvam.ai/text-to-speech/ws"


class SarvamClient(TTSProviderClient):
    def __init__(
        self,
        *,
        api_key: Optional[str],
        voice: str,
        language: str = "en-IN",
        sample_rate: int = 24000,
        model: str = "bulbul:v2",
        url: str = SARVAM_DEFAULT_URL,
    ) -> None:
        self.sample_rate = sample_rate
        self.num_channels = 1
        self._api_key = api_key
        self._voice = voice
        self._language = language
        self._model = model
        self._url = url
        self._ws: Optional[Any] = None

    async def connect(self) -> None:
        self._ws = await websockets.connect(
            self._url,
            additional_headers={"api-subscription-key": self._api_key or ""},
        )
        await self._ws.send(
            json.dumps(
                {
                    "type": "config",
                    "data": {
                        "target_language_code": self._language,
                        "speaker": self._voice,
                        "model": self._model,
                        "speech_sample_rate": self.sample_rate,
                        "output_audio_codec": "linear16",
                        "min_buffer_size": 50,
                        "max_chunk_length": 200,
                        "send_completion_event": True,
                    },
                }
            )
        )

    async def send_text(self, text: str) -> None:
        if self._ws is None:
            raise TTSConnectionError("Sarvam client used before connect()")
        await self._ws.send(json.dumps({"type": "text", "data": {"text": text}}))

    async def flush(self) -> None:
        if self._ws is None:
            raise TTSConnectionError("Sarvam client used before connect()")
        await self._ws.send(json.dumps({"type": "flush"}))

    async def close(self) -> None:
        if self._ws is None:
            return
        with contextlib.suppress(Exception):
            await self._ws.close()
        self._ws = None

    async def audio_chunks(self) -> AsyncIterator[bytes]:
        if self._ws is None:
            raise TTSConnectionError("Sarvam client used before connect()")
        async for raw in self._ws:
            msg = json.loads(raw)
            msg_type = msg.get("type")
            if msg_type == "audio":
                b64_audio = msg.get("data", {}).get("audio")
                if b64_audio:
                    yield base64.b64decode(b64_audio)
            elif msg_type == "error":
                raise TTSProviderError(
                    msg.get("data", {}).get("message", "unknown Sarvam TTS error")
                )
            # "event"/"final" completion notices are informational only —
            # the connection stays open for the next turn.


def build_provider_client(
    provider: str, config: TTSConfig, *, api_key_override: Optional[str] = None
) -> TTSProviderClient:
    api_key = api_key_override or config.api_key
    if provider == "fish":
        return FishAudioClient(api_key=api_key, voice=config.voice, sample_rate=config.sample_rate)
    if provider == "sarvam":
        return SarvamClient(
            api_key=api_key,
            voice=config.voice,
            language=config.language,
            sample_rate=config.sample_rate,
        )
    raise ValueError(f"unknown TTS provider: {provider!r}")


# ── Connection lifecycle (retry + fallback) ──────────────────────────────────


async def _connect_with_backoff(
    make_client: Callable[[], TTSProviderClient],
    *,
    max_retries: int = 4,
    base_delay: float = 0.25,
) -> TTSProviderClient:
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        client = make_client()
        try:
            await client.connect()
            return client
        except Exception as exc:  # noqa: BLE001 — any connect failure is retryable
            last_exc = exc
            delay = base_delay * (2**attempt)
            logger.warning(
                "[tts] connect attempt %d/%d failed: %s — retrying in %.2fs",
                attempt + 1,
                max_retries,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
    raise TTSConnectionError(f"exhausted {max_retries} connect attempts") from last_exc


async def _connect_provider(config: TTSConfig) -> TTSProviderClient:
    try:
        return await _connect_with_backoff(lambda: build_provider_client(config.provider, config))
    except TTSConnectionError:
        if not config.fallback_provider or config.fallback_provider == config.provider:
            raise
        logger.error(
            "[tts] primary provider %r unreachable — falling back to %r",
            config.provider,
            config.fallback_provider,
        )
        fallback_key = config.fallback_api_key or config.api_key
        return await _connect_with_backoff(
            lambda: build_provider_client(
                config.fallback_provider, config, api_key_override=fallback_key
            )
        )


# ── Worker ────────────────────────────────────────────────────────────────────


def _drain_queue(queue: "asyncio.Queue[Any]") -> int:
    """Discard every currently-pending item without blocking."""
    count = 0
    while True:
        try:
            queue.get_nowait()
            count += 1
        except asyncio.QueueEmpty:
            return count


async def _pump_audio(client: TTSProviderClient, audio_out_queue: "asyncio.Queue[bytes]") -> None:
    async for chunk in client.audio_chunks():
        await audio_out_queue.put(chunk)


async def _run_session(
    client: TTSProviderClient,
    text_queue: "asyncio.Queue[Any]",
    audio_out_queue: "asyncio.Queue[bytes]",
    interruption_event: asyncio.Event,
    clause_buf: ClauseBuffer,
) -> str:
    """Run one WebSocket session: pump text in, pump audio out, watch for
    barge-in. Returns "shutdown", "interrupted", or "dropped" so the caller
    knows whether to stop, purge+reconnect, or just reconnect."""
    audio_task = asyncio.ensure_future(_pump_audio(client, audio_out_queue))
    interrupt_task = asyncio.ensure_future(interruption_event.wait())
    try:
        while True:
            get_task = asyncio.ensure_future(text_queue.get())
            done, _pending = await asyncio.wait(
                {get_task, audio_task, interrupt_task}, return_when=asyncio.FIRST_COMPLETED
            )

            if interrupt_task in done:
                get_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await get_task
                return "interrupted"

            if audio_task in done:
                get_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await get_task
                exc = audio_task.exception()
                if exc:
                    logger.warning("[tts] audio stream dropped: %s", exc)
                else:
                    logger.warning("[tts] audio stream ended unexpectedly")
                return "dropped"

            item = get_task.result()
            if item is SHUTDOWN:
                return "shutdown"
            if item is TURN_END:
                trailing = clause_buf.drain()
                if trailing:
                    await client.send_text(trailing)
                await client.flush()
                continue

            for clause in clause_buf.push(item):
                await client.send_text(clause)
                await client.flush()
    finally:
        for task in (audio_task, interrupt_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(audio_task, interrupt_task, return_exceptions=True)


async def stream_tts_worker(
    text_queue: "asyncio.Queue[Any]",
    audio_out_queue: "asyncio.Queue[Any]",
    interruption_event: asyncio.Event,
    *,
    config: Optional[TTSConfig] = None,
    connect: Optional[Callable[[], Coroutine[Any, Any, TTSProviderClient]]] = None,
) -> None:
    """Consume LLM tokens from ``text_queue``, stream them to the configured
    TTS provider over a persistent WebSocket, and push raw audio bytes onto
    ``audio_out_queue`` as they arrive.

    Push ``TURN_END`` onto text_queue when one LLM turn's tokens are done
    (flushes the trailing clause, connection stays open). Push ``SHUTDOWN``
    to stop the worker for good (closes the connection and returns).

    ``connect`` is an injectable ``() -> TTSProviderClient`` factory, mainly
    for tests; defaults to the real env-configured provider with retry +
    fallback.
    """
    config = config or TTSConfig.from_env()
    connect = connect or (lambda: _connect_provider(config))

    clause_buf = ClauseBuffer()
    client = await connect()
    logger.info("[tts] connected via %s", type(client).__name__)

    try:
        while True:
            outcome = await _run_session(
                client, text_queue, audio_out_queue, interruption_event, clause_buf
            )

            if outcome == "shutdown":
                break

            if outcome == "interrupted":
                logger.info("[tts] barge-in — purging audio buffer and resetting stream")
                _drain_queue(audio_out_queue)
                _drain_queue(text_queue)
                clause_buf.reset()
                interruption_event.clear()

            # Both "interrupted" and "dropped" leave the socket in a state we
            # don't trust — close it and open a fresh one before resuming.
            await client.close()
            client = await connect()
            logger.info("[tts] reconnected via %s (reason=%s)", type(client).__name__, outcome)
    finally:
        await client.close()
