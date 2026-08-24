# Nexus (LiveKit + RAG Voice Agent)

An end-to-end real-time voice agent built with LiveKit, FastAPI, and Next.js. The agent is capable of real-time voice conversations and utilizes Retrieval-Augmented Generation (RAG) to answer questions based on uploaded PDF documents.

## Real-Time Telemetry Matrix

End-to-end latency budget from voice-in to audio-out. Each stage is independently monitored and has a hard P95 target.

| Stage | Component | P50 Target | P95 Target | Notes |
|---|---|---|---|---|
| **VAD** | Silero VAD | 20 ms | 40 ms | Local inference; no network hop |
| **STT** | Deepgram Nova-2 | 80 ms | 120 ms | Streaming partial transcripts |
| **LLM** | Groq (llama-3.3-70b) | 120 ms | 180 ms | Time-to-first-token via Groq LPU |
| **TTS** | Fish Audio / Sarvam (WS) | 90 ms | 150 ms | Direct WebSocket stream; first chunk |
| **Network** | LiveKit WebRTC | 20 ms | 45 ms | Regional relay; TURN fallback |
| **Total** | Full pipeline | **330 ms** | **~535 ms** | Barge-in cuts this to VAD+Network |

> **Barge-in:** When the user interrupts, the pipeline short-circuits at VAD + Network (~65 ms). LLM generation is cancelled via `session.interrupt()`; the TTS worker purges its audio/text queues and clears the LiveKit audio source's playout buffer (`AudioSource.clear_queue()`), then reconnects — neither Fish Audio nor Sarvam expose an in-place "cancel this utterance" message.

## Tech Stack

- **WebRTC/Orchestration:** LiveKit
- **STT:** Deepgram
- **LLM:** Any OpenAI-compatible endpoint via `LLM_BASE_URL`/`LLM_MODEL` (Groq, Cerebras, etc. — see [Real-Time Telemetry Matrix](#real-time-telemetry-matrix))
- **TTS:** Direct WebSocket streaming via `TTS_PROVIDER`/`TTS_VOICE` (Fish Audio or Sarvam — see [Streaming TTS Layer](#streaming-tts-layer))
- **Vector Store:** ChromaDB (In-Memory)
- **Backend:** FastAPI (Python)
- **Frontend:** Next.js 15 (React) + Tailwind

## Environment Variables

Create a `.env` file in the root (or `backend/`) directory based on `.env.example`:

```
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
DEEPGRAM_API_KEY=your_deepgram_key

# LLM inference — any OpenAI-compatible chat completions endpoint
LLM_BASE_URL=https://api.groq.com/openai/v1   # or https://api.cerebras.ai/v1
LLM_API_KEY=your_llm_provider_key
LLM_MODEL=llama-3.3-70b-versatile

# Streaming TTS — direct WebSocket connection to Fish Audio or Sarvam
TTS_PROVIDER=fish                              # or sarvam
TTS_API_KEY=your_tts_provider_key
TTS_VOICE=default
TTS_SAMPLE_RATE=24000
```

### LLM Inference Layer

The voice worker's LLM is decoupled from any specific provider — `backend/llm_provider.py`
builds the LiveKit `openai.LLM` plugin from `LLM_BASE_URL`/`LLM_API_KEY`/`LLM_MODEL`, so
swapping Groq for Cerebras (or any other OpenAI-compatible endpoint) is a config change,
not a code change.

`backend/llm_stream.py` provides a standalone, provider-agnostic streaming primitive built
on `AsyncOpenAI`: tokens are pushed onto an `asyncio.Queue` the instant each delta arrives
on the wire (no sentence buffering), and `backend/latency.py` records **TTFT** (time to
first token) and **output TPS** for every stream. Benchmark the configured provider directly:

```bash
cd backend
python bench_llm.py "What is the capital of France?"
```

Unit tests for the streaming layer (mocked provider, no network calls):

```bash
cd backend
python -m unittest test_llm_stream -v
```

### Streaming TTS Layer

`backend/tts_streamer.py` connects directly to a provider's WebSocket streaming API —
Fish Audio (`wss://api.fish.audio/v1/tts/live`, MessagePack framing) or Sarvam
(`wss://api.sarvam.ai/text-to-speech/ws`, JSON framing) — instead of a batch/REST TTS
call. `stream_tts_worker(text_queue, audio_out_queue, interruption_event)` buffers
incoming LLM tokens into clauses (flushed on `, . ? ! \n`), streams each clause into a
persistent WebSocket connection, and pushes raw PCM16 audio bytes onto `audio_out_queue`
the instant they arrive — no sentence-level batching on the way out either. On barge-in
(`interruption_event`) it purges both queues and reconnects, since neither provider
exposes an in-place "cancel this utterance" message. A socket drop mid-stream triggers
exponential-backoff reconnect, falling back to `TTS_FALLBACK_PROVIDER` if the primary
stays unreachable.

`backend/livekit_audio_publisher.py` is the consumer side: it re-chunks the provider's
bursty byte deliveries into uniform frames for `livekit.rtc.AudioSource`, whose own
internal playout buffer (`queue_size_ms`) absorbs network/provider jitter without
stuttering, and calls `AudioSource.clear_queue()` on barge-in to drop already-buffered
audio immediately.

Unit tests (in-memory fake provider client, no network calls):

```bash
cd backend
python -m unittest test_tts_streamer -v
```

### Turn-Taking Arbiter

`backend/turn_arbiter.py` addresses two false-positive turn-taking triggers: a user
backchannel ("yeah", "uh-huh", "okay") getting mistaken for a real turn, and the user
pausing mid-thought on a conjunction or preposition ("...and", "because", "my...")
triggering endpointing before they're actually done.

`RollingTranscriptBuffer` keeps the last 3-5s of Deepgram interim/final transcripts
(timestamp, confidence, terminal-punctuation per event). `classify_turn_intent(transcript)`
is dual-stage: a regex fast path resolves common backchannels, dangling-connector pauses,
and obviously-complete sentences in well under a millisecond with zero network calls;
anything genuinely ambiguous escalates to a bounded-timeout (`TURN_ARBITER_TIMEOUT_MS`,
default 100ms) structured-JSON call against the same fast LLM client from `llm_stream.py`.
A timeout, provider error, or malformed response never raises — it falls back to the same
regex heuristics, so a slow/dead provider can't stall turn-taking. `TurnArbiter` ties the
two together (`await arbiter.evaluate(text, is_final=...)`). Not yet wired into `agent.py`'s
STT handling — it's a standalone, tested module ready to plug into
`on_user_input_transcribed`.

Unit tests (mocked LLM client, no network calls):

```bash
cd backend
python -m unittest test_turn_arbiter -v
```

## Setup & Run Instructions (Local)

### 1. Start the Backend API (FastAPI)

Handles token generation and RAG document ingestion.

```bash
cd backend
python -m venv .venv
source .venv/Scripts/activate  # Windows
# source .venv/bin/activate    # macOS/Linux
pip install -r requirements.txt
python main.py
```

### 2. Start the Voice Worker (LiveKit Agent)

The background worker that orchestrates the STT → LLM → TTS pipeline.

```bash
cd backend
source .venv/Scripts/activate
python agent.py start
```

### 3. Start the Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Setup & Run Instructions (Docker)

To run the entire stack via Docker Compose:

```bash
docker-compose up --build
```

This starts:
- **backend** (FastAPI API) on port 8000
- **agent** (LiveKit voice worker) — connects to LiveKit Cloud
- **frontend** (Next.js) on port 3000

> **Note:** The LiveKit agent worker (`agent.py`) connects to LiveKit Cloud directly and does not expose a local port.

## How to Use

1. **Upload a Document** — Upload a PDF via the UI. The document is chunked and stored in the in-memory vector store.
2. **Tweak the System Prompt** — Edit the system prompt to customize agent behavior before connecting.
3. **Connect to Agent** — Click "Connect to Agent" to join a LiveKit room. Allow microphone access.
4. **Ask Questions** — Talk to the agent. Ask about the uploaded document. The "RAG Sources" panel will show knowledge base queries as they occur, and the live transcript will display agent responses.

## Known Limitations & Tradeoffs

- **Persistent Vector Store:** ChromaDB uses a `PersistentClient` (stored in `backend/chroma_data/`). Uploaded documents survive backend restarts, but all documents share a single collection — there is no per-session or per-user isolation.
- **Native Chunking:** The RAG ingestion relies on a custom, pure-Python character-overlap chunker. While fast and minimizing dependencies, it does not currently use semantic boundary detection.
- **Single Collection:** All uploaded documents share a single ChromaDB collection with no per-session or per-document isolation. Clearing the KB requires deleting `backend/chroma_data/`.
- **CORS:** CORS is open (`*`) for ease of local and Docker development. Restrict origins before any production deployment.
