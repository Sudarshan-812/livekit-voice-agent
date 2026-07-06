# FinOps — Token & Cost Guardrail Layer

Operational cost controls for Nexus. Every API call that leaves the process boundary
has a corresponding budget gate, usage counter, and circuit-breaker.

---

## 1. Cost Taxonomy

| Resource | Unit | Provider | Est. Rate |
|---|---|---|---|
| STT | per audio-minute | Deepgram | $0.0059 / min |
| LLM prompt | per 1K tokens | Groq | $0.00059 / 1K |
| LLM completion | per 1K tokens | Groq | $0.00079 / 1K |
| TTS | per 1K characters | Deepgram | $0.0150 / 1K |
| WebRTC relay | per participant-minute | LiveKit Cloud | see plan |

---

## 2. Per-Session Budget Caps

Each `JobContext` session gets a hard cap on token spend. Exceeding the cap
triggers a graceful shutdown via the `IdleWatchdog` disconnect path.

```
# TODO: implement in track_cost()
SESSION_MAX_PROMPT_TOKENS  = 50_000
SESSION_MAX_AUDIO_MINUTES  = 10.0
SESSION_MAX_TTS_CHARS      = 25_000
```

---

## 3. Redis Cost Tracking Schema

All counters are stored in Redis with a TTL equal to the session's max lifetime.

```
nexus:tokens:prompt:{session_id}     → INT   (prompt tokens consumed)
nexus:tokens:completion:{session_id} → INT   (completion tokens consumed)
nexus:audio:secs:{session_id}        → FLOAT (STT audio seconds billed)
nexus:tts:chars:{session_id}         → INT   (TTS characters synthesized)
nexus:cost:usd:{session_id}          → FLOAT (running USD estimate)
```

Implementation entry point: `async def track_cost(session_id, prompt_tokens, audio_seconds)` in `backend/agent.py`.

---

## 4. Idle Circuit Breaker

The `IdleWatchdog` in `agent.py` hard-disconnects WebRTC after **45 seconds** of
silence (no user speech, no agent speech). This eliminates "ghost sessions" that
burn STT streaming costs with no user present.

```
Configurable via env:
  NEXUS_IDLE_TIMEOUT_SECONDS=45  (default)
```

---

## 5. Aggregate Spend Dashboard (TODO)

- [ ] Grafana dashboard pulling from Redis counters via redis-exporter
- [ ] Daily spend alert → PagerDuty / Slack webhook at $5 threshold
- [ ] Per-model spend breakdown (Groq vs Deepgram vs LiveKit)
- [ ] Cost-per-session histogram for capacity planning

---

## 6. Model Tiering Strategy (TODO)

| Tier | Trigger | LLM | TTS | Expected Saving |
|---|---|---|---|---|
| Premium | Default | llama-3.3-70b | Deepgram Aura-2 | baseline |
| Standard | > 30 sessions/hr | llama-3.1-8b | Deepgram Aura | ~60% |
| Fallback | Budget cap hit | llama-3.1-8b | Deepgram Basic | ~80% |
