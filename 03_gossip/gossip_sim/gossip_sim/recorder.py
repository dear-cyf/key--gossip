"""Light-weight recorder that works with all three simulators.

The stage-1 and stage-2 simulators call a *recorder* object with event hooks.
This recorder duck-types their interface and collects only what the gossip
comparison needs:

* raw end-to-end latency of **every** delivered update
  (``t_delivery - msg.emit_t``), and of the **first** delivery per
  ``(src, dst)`` pair -- the metric stage 3 measured on the real cluster;
* write/message counters per arm (a batch counts as one write, K messages);
* received/stale counters per arm.

Nothing is sampled: for N=1000 the latency list is ~3.6M float32 (14 MB).
"""

from __future__ import annotations

from array import array
from typing import Any, Dict, Optional


class MetricRecorder:
    trace_nodes = ()  # the sims read this via getattr(self.rec, "trace_nodes", ())

    def __init__(self, num_nodes: int, collect_latency: bool = True):
        self.n = int(num_nodes)
        self.collect_latency = bool(collect_latency)
        self.all_lat = array("f")      # every accepted delivery
        self.first_lat = array("f")    # first delivery per (src, dst)
        self.first_pair_seen = bytearray(self.n * self.n)
        self.all_lat_w1 = array("f")
        self.all_lat_w2 = array("f")
        self.deliveries = 0
        self.writes = 0
        self.messages_written = 0
        self.batch_sizes: Dict[int, int] = {}
        self.max_batch = 0
        self.emits = 0
        self.first_wave2_emit_t: Optional[float] = None
        self.ae_deliveries = 0

    # ------------------------------------------------------------------ #
    # hooks used by the three simulators
    # ------------------------------------------------------------------ #
    def on_emit(self, t: float, msg) -> None:  # noqa: D401
        self.emits += 1
        if getattr(msg, "wave", 1) == 2 and self.first_wave2_emit_t is None:
            self.first_wave2_emit_t = t

    def on_gcs_in(self, t: float, msg, accepted: bool, q_len: int) -> None:
        pass

    def on_fanout(self, start_t: float, end_t: float, msg) -> None:
        pass

    def on_write(self, *args) -> None:
        """stage1: (t, dst, msg, enq_t, start_t, delivered, backlog)
        stage2: (t, dst, batch, start_t, delivered, trigger)"""
        payload = args[2]
        self.writes += 1
        if isinstance(payload, list):
            n = len(payload)
            self.messages_written += n
            self.batch_sizes[n] = self.batch_sizes.get(n, 0) + 1
            self.max_batch = max(self.max_batch, n)
        else:
            self.messages_written += 1
            self.batch_sizes[1] = self.batch_sizes.get(1, 0) + 1
            self.max_batch = max(self.max_batch, 1)

    def on_recv(self, t: float, dst: int, msg) -> None:
        self.deliveries += 1
        if getattr(msg, "wave", 1) == 0:
            # anti-entropy repair, not a fresh-update propagation
            self.ae_deliveries += 1
            return
        if not self.collect_latency:
            return
        lat = (t - msg.emit_t) * 1000.0  # ms
        self.all_lat.append(lat)
        if getattr(msg, "wave", 1) == 1:
            self.all_lat_w1.append(lat)
        else:
            self.all_lat_w2.append(lat)
        key = msg.src * self.n + dst
        if not self.first_pair_seen[key]:
            self.first_pair_seen[key] = 1
            self.first_lat.append(lat)

    def on_sample(self, t: float, gauges: Dict[str, Any]) -> None:
        pass

    # ------------------------------------------------------------------ #
    def stats(self, pct_fn) -> Dict[str, Optional[float]]:
        def q(arr, p):
            return None if len(arr) == 0 else float(pct_fn(arr, p))

        all_l = self.all_lat
        first_l = self.first_lat
        mean = (sum(all_l) / len(all_l)) if all_l else None
        mean_first = (sum(first_l) / len(first_l)) if first_l else None
        return {
            "deliveries": self.deliveries,
            "deliveries_w1": len(self.all_lat_w1),
            "deliveries_w2": len(self.all_lat_w2),
            "lat_all_mean_ms": mean,
            "lat_all_p50_ms": q(all_l, 50),
            "lat_all_p95_ms": q(all_l, 95),
            "lat_all_p99_ms": q(all_l, 99),
            "lat_all_max_ms": (max(all_l) if all_l else None),
            "lat_w1_mean_ms": (sum(self.all_lat_w1) / len(self.all_lat_w1)
                               if self.all_lat_w1 else None),
            "lat_w1_p95_ms": q(self.all_lat_w1, 95),
            "lat_w2_mean_ms": (sum(self.all_lat_w2) / len(self.all_lat_w2)
                               if self.all_lat_w2 else None),
            "lat_w2_p95_ms": q(self.all_lat_w2, 95),
            "lat_first_pair_mean_ms": mean_first,
            "lat_first_pair_p95_ms": q(first_l, 95),
            "lat_first_pair_p99_ms": q(first_l, 99),
            "write_events": self.writes,
            "messages_written": self.messages_written,
            "updates_emitted": self.emits,
            "max_batch_in_recorder": (self.max_batch or None),
            "wave2_first_emit_s": self.first_wave2_emit_t,
        }
