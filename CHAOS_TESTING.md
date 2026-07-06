# Chaos Testing — Load Testing Methodology & Fault Injection

Resilience playbook for Nexus. Each scenario maps to a known failure mode observed in
production voice AI deployments, with a pass/fail acceptance criterion.

---

## 1. Load Testing Methodology

### 1.1 Tooling

| Tool | Purpose |
|---|---|
| `locust` | HTTP endpoint load (token API, RAG ingestion) |
| `livekit-cli` | Synthetic WebRTC participant load |
| `k6` | Latency percentile profiling under concurrent load |
| `wrk` | Burst throughput on `/token` endpoint |

### 1.2 Target Scenarios

| Scenario | Concurrent Sessions | Duration | Pass Criterion |
|---|---|---|---|
| Baseline | 1 | 5 min | P95 E2E latency ≤ 535 ms |
| Ramp | 1 → 10 | 10 min | No session drops; P95 ≤ 700 ms |
| Soak | 5 | 60 min | Memory stable (±5 MB); no leaked tasks |
| Spike | 0 → 20 in 10s | 5 min | Graceful queuing; no 5xx |
| Barge-in storm | 1 session, interrupt every 2s | 3 min | No deadlocks; CPU ≤ 80% |

---

## 2. Fault Injection Matrix

### 2.1 Network Faults

```bash
# Simulate 200ms RTT jitter on LiveKit WebRTC path (Linux tc)
tc qdisc add dev eth0 root netem delay 200ms 50ms distribution normal

# Simulate 5% packet loss
tc qdisc change dev eth0 root netem loss 5%

# Reset
tc qdisc del dev eth0 root
```

### 2.2 Upstream API Faults

| Fault | Injection Method | Expected Behavior |
|---|---|---|
| Deepgram STT timeout | toxiproxy latency on port 443 | STT falls back; session logs error |
| Groq 429 rate limit | mock server returning 429 | LLM retries with backoff; user hears hold message |
| Deepgram TTS failure | toxiproxy reset | Agent logs error; session continues (text only) |
| LiveKit TURN failure | block TURN ports in firewall | Direct ICE fails; agent logs warning |

### 2.3 Process-Level Faults

```bash
# Kill the agent worker mid-session (tests LiveKit reconnect)
kill -9 $(pgrep -f "agent.py")

# OOM simulation (tests asyncio task cleanup)
stress --vm 1 --vm-bytes 900M --timeout 30s
```

---

## 3. Idle Watchdog Verification

```bash
# Join a session and stay silent — watchdog should disconnect after 45s
python tests/chaos/silent_participant.py --timeout 60

# Expected log output:
# [watchdog] 45.0s idle — forcibly disconnecting to purge API costs
```

---

## 4. Barge-in Concurrency Test

```bash
# Fire rapid barge-in signals to verify no asyncio deadlock
python tests/chaos/barge_in_storm.py --interval 0.5 --duration 60
```

Pass criterion: `interrupt_event` fires and clears cleanly every cycle. No `asyncio.Task` leak detected by `asyncio.all_tasks()` growth.

---

## 5. Memory Leak Validation

```bash
# Run with tracemalloc to detect task/buffer leaks
PYTHONTRACEMALLOC=10 python agent.py start
```

Check after soak test:
- `asyncio.all_tasks()` count returns to baseline between sessions
- `InterruptController._active_tasks` set is empty between utterances
- No growing `bytes` allocation in the LiveKit RTC buffer pool

---

## 6. Acceptance Gate (CI)

- [ ] Baseline load test passes in CI (`locust --headless --users 1 --run-time 2m`)
- [ ] Barge-in storm runs 60s without deadlock
- [ ] Silent participant disconnects within 50s (watchdog fires ≤ 45s + 5s grace)
- [ ] No asyncio warnings in log output (`Task was destroyed but it is pending!`)
- [ ] P95 E2E latency ≤ 535 ms under baseline load
