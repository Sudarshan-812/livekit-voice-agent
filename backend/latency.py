"""Latency benchmarking for LLM token streaming — TTFT and output TPS."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("nexus.llm.latency")


@dataclass
class StreamLatency:
    """Tracks time-to-first-token and throughput for one streamed completion.

    TTFT is measured from construction (i.e. immediately before the
    provider request is issued) to the first token delta observed. TPS is
    measured over the generation window only (first token → last token),
    since that isolates decode throughput from network/queue TTFT.
    """

    request_id: str
    model: str
    _start: float = field(default_factory=time.perf_counter, init=False)
    _first_token_at: Optional[float] = field(default=None, init=False)
    token_count: int = field(default=0, init=False)

    def record_token(self) -> None:
        """Call once per token/delta yielded to the consumer."""
        now = time.perf_counter()
        if self._first_token_at is None:
            self._first_token_at = now
            logger.info(
                "[llm-latency] request=%s model=%s ttft_ms=%.1f",
                self.request_id,
                self.model,
                (now - self._start) * 1000,
            )
        self.token_count += 1

    def finish(self) -> dict:
        """Call once the stream is exhausted. Logs and returns the summary."""
        end = time.perf_counter()
        ttft_ms = (
            (self._first_token_at - self._start) * 1000
            if self._first_token_at is not None
            else None
        )
        gen_s = (end - self._first_token_at) if self._first_token_at is not None else 0.0
        tps = (self.token_count / gen_s) if gen_s > 0 else 0.0

        summary = {
            "request_id": self.request_id,
            "model": self.model,
            "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
            "tokens": self.token_count,
            "total_s": round(end - self._start, 3),
            "tps": round(tps, 1),
        }
        logger.info(
            "[llm-latency] request=%s model=%s ttft_ms=%s tokens=%d tps=%.1f total_s=%.3f",
            summary["request_id"],
            summary["model"],
            summary["ttft_ms"],
            summary["tokens"],
            summary["tps"],
            summary["total_s"],
        )
        return summary
