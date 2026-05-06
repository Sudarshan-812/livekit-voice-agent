# Real-Time Voice AI Agent (LiveKit + RAG)

An end-to-end real-time voice agent built with LiveKit, FastAPI, and Next.js. The agent is capable of real-time voice conversations and utilizes Retrieval-Augmented Generation (RAG) to answer questions based on uploaded PDF documents.

## Tech Stack

- **WebRTC/Orchestration:** LiveKit
- **STT:** Deepgram
- **LLM:** Gemini 1.5 Flash
- **TTS:** Google TTS
- **Vector Store:** ChromaDB (In-Memory)
- **Backend:** FastAPI (Python)
- **Frontend:** Next.js 15 (React) + Tailwind

## Environment Variables

Create a `.env` file in the root (or `backend/`) directory based on `.env.example`:

```
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
GEMINI_API_KEY=your_gemini_key
DEEPGRAM_API_KEY=your_deepgram_key
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

- **In-Memory Vector Store:** ChromaDB is configured as an ephemeral, in-memory client. Uploaded documents do not persist across backend restarts.
- **Native Chunking:** The RAG ingestion relies on a custom, pure-Python character-overlap chunker. While fast and minimizing dependencies, it does not currently use semantic boundary detection.
- **Single Collection:** All uploaded documents share a single ChromaDB collection. There is no per-session or per-document isolation.
- **CORS:** CORS is open (`*`) for ease of local and Docker development. Restrict origins before any production deployment.
