"""Monitoring / measurement layer for the Ray Syncer simulation.

Records exactly the quantities the monitoring requirement asks for:

* updates produced by the Raylets per second,
* messages the GCS receives per second,
* messages the GCS broadcasts (fans out) and writes per second,
* average per-node send / receive rate per second,
* queue length (GCS inbound backlog, per-connection ``sending_buffer_``),
* propagation latency per message (emit -> local view updated),
* convergence time.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import SimConfig

DETAIL_FIELDS = [
    "mid", "src", "dst", "wave", "round", "emit_t", "gcs_recv_t", "enqueue_t",
    "write_start_t", "delivered_t", "backlog_at_enqueue", "gcs_in_backlog_at_recv",
    "gcs_queue_delay_s", "link_queue_wait_s", "propagation_latency_s",
    "end_to_end_latency_s",
]


def _pct(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


class Recorder:
    """Buckets every interesting event into one-second slices."""

    def __init__(self, cfg: SimConfig, num_nodes: int):
        self.cfg = cfg
        self.N = num_nodes
        if num_nodes <= cfg.detail_max_nodes:
            self.detail_dsts = set(range(num_nodes))
        else:
            k = max(1, cfg.detail_sample_nodes)
            step = max(1, num_nodes // k)
            self.detail_dsts = set(range(0, num_nodes, step))
        self.trace_nodes = tuple(sorted(self.detail_dsts))[:8]

        self.sec: Dict[int, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.gauges: Dict[int, Dict[str, Any]] = {}
        self.detail: List[Tuple] = []

        # Global per-wave aggregates (all messages, not only sampled links).
        self.agg: Dict[int, Dict[str, float]] = {
            1: defaultdict(float),
            2: defaultdict(float),
        }
        self.first_wave2_emit_t: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Event hooks
    # ------------------------------------------------------------------ #
    def _bucket(self, t: float) -> Dict[str, float]:
        return self.sec[int(math.floor(max(0.0, t)))]

    def on_emit(self, t: float, msg) -> None:
        b = self._bucket(t)
        b["updates_emitted"] += 1
        b[f"updates_emitted_w{msg.wave}"] += 1
        if msg.wave == 2 and self.first_wave2_emit_t is None:
            self.first_wave2_emit_t = t

    def on_gcs_in(self, t: float, msg, accepted: bool, q_len: int) -> None:
        b = self._bucket(t)
        b["gcs_received"] += 1
        b["gcs_in_backlog_at_recv_sum"] += q_len
        if accepted:
            b["gcs_accepted"] += 1
            b[f"gcs_accepted_w{msg.wave}"] += 1
            if q_len > b["gcs_in_backlog_at_recv_max"]:
                b["gcs_in_backlog_at_recv_max"] = q_len
        else:
            b["gcs_stale_dropped"] += 1

    def on_fanout(self, start_t: float, end_t: float, msg) -> None:
        b = self._bucket(end_t)
        b["gcs_broadcasts"] += 1
        b[f"gcs_broadcasts_w{msg.wave}"] += 1
        b["gcs_busy_s"] += end_t - start_t
        if msg.gcs_recv_t is not None:
            delay = start_t - msg.gcs_recv_t
            b["gcs_queue_delay_sum"] += delay
            b["gcs_queue_delay_cnt"] += 1
            w = msg.wave
            self.agg[w]["gcsq_sum"] += delay
            self.agg[w]["gcsq_cnt"] += 1
            self.agg[w]["gcsq_max"] = max(self.agg[w]["gcsq_max"], delay)

    def on_write(self, t: float, dst: int, msg, enq_t: float, start_t: float,
                 delivered_t: float, backlog_at_enq: int) -> None:
        b = self._bucket(delivered_t)
        b["link_writes"] += 1
        b[f"link_writes_w{msg.wave}"] += 1
        wait = start_t - enq_t
        e2e = delivered_t - msg.emit_t
        b["link_queue_wait_sum"] += wait
        b["link_queue_wait_cnt"] += 1
        b["propagation_latency_sum"] += e2e
        b["propagation_latency_cnt"] += 1
        b["backlog_at_enqueue_sum"] += backlog_at_enq
        b["backlog_at_enqueue_cnt"] += 1
        if backlog_at_enq > b["backlog_at_enqueue_max"]:
            b["backlog_at_enqueue_max"] = backlog_at_enq
        w = msg.wave
        self.agg[w]["wait_sum"] += wait
        self.agg[w]["wait_cnt"] += 1
        self.agg[w]["e2e_sum"] += e2e
        self.agg[w]["e2e_cnt"] += 1
        self.agg[w]["e2e_max"] = max(self.agg[w]["e2e_max"], e2e)
        self.agg[w]["backlog_sum"] += backlog_at_enq
        self.agg[w]["backlog_cnt"] += 1
        self.agg[w]["backlog_max"] = max(self.agg[w]["backlog_max"], backlog_at_enq)

        if dst in self.detail_dsts:
            self.detail.append((
                msg.mid, msg.src, dst, msg.wave, msg.rnd, msg.emit_t,
                msg.gcs_recv_t if msg.gcs_recv_t is not None else float("nan"),
                enq_t, start_t, delivered_t, backlog_at_enq, msg.gcs_in_q_at,
                (start_t - msg.gcs_recv_t) if msg.gcs_recv_t is not None else float("nan"),
                wait, e2e, e2e,
            ))

    def on_recv(self, t: float, dst: int, msg) -> None:
        b = self._bucket(t)
        b["node_received"] += 1

    def on_sample(self, t: float, gauges: Dict[str, Any]) -> None:
        key = int(math.floor(max(0.0, t)))
        if key in self.gauges:
            return
        self.gauges[key] = dict(gauges)

    # ------------------------------------------------------------------ #
    # Exports
    # ------------------------------------------------------------------ #
    def per_second_records(self, num_nodes: int) -> List[Dict[str, Any]]:
        if not self.sec and not self.gauges:
            return []
        last = max([0] + list(self.sec.keys()) + list(self.gauges.keys()))
        rows: List[Dict[str, Any]] = []
        for s in range(0, last + 1):
            b = self.sec.get(s, {})
            g = self.gauges.get(s, {})
            rec = {
                "second": s,
                "updates_emitted": b.get("updates_emitted", 0),
                "updates_emitted_w1": b.get("updates_emitted_w1", 0),
                "updates_emitted_w2": b.get("updates_emitted_w2", 0),
                "gcs_received": b.get("gcs_received", 0),
                "gcs_accepted": b.get("gcs_accepted", 0),
                "gcs_accepted_w1": b.get("gcs_accepted_w1", 0),
                "gcs_accepted_w2": b.get("gcs_accepted_w2", 0),
                "gcs_stale_dropped": b.get("gcs_stale_dropped", 0),
                "gcs_broadcasts": b.get("gcs_broadcasts", 0),
                "gcs_broadcasts_w1": b.get("gcs_broadcasts_w1", 0),
                "gcs_broadcasts_w2": b.get("gcs_broadcasts_w2", 0),
                "link_writes": b.get("link_writes", 0),
                "link_writes_w1": b.get("link_writes_w1", 0),
                "link_writes_w2": b.get("link_writes_w2", 0),
                "node_received": b.get("node_received", 0),
                "avg_node_sent_per_sec": b.get("updates_emitted", 0) / num_nodes,
                "avg_node_recv_per_sec": b.get("node_received", 0) / num_nodes,
                "gcs_in_backlog_at_recv_avg": (
                    b.get("gcs_in_backlog_at_recv_sum", 0.0)
                    / b.get("gcs_received", 1)
                ),
                "gcs_in_backlog_at_recv_max": b.get("gcs_in_backlog_at_recv_max", 0),
                "gcs_queue_delay_avg_ms": (
                    1000.0 * b.get("gcs_queue_delay_sum", 0.0)
                    / b.get("gcs_queue_delay_cnt", 1)
                ),
                "link_queue_wait_avg_ms": (
                    1000.0 * b.get("link_queue_wait_sum", 0.0)
                    / b.get("link_queue_wait_cnt", 1)
                ),
                "backlog_at_enqueue_avg": (
                    b.get("backlog_at_enqueue_sum", 0.0)
                    / b.get("backlog_at_enqueue_cnt", 1)
                ),
                "backlog_at_enqueue_max": b.get("backlog_at_enqueue_max", 0),
                "propagation_latency_avg_ms": (
                    1000.0 * b.get("propagation_latency_sum", 0.0)
                    / b.get("propagation_latency_cnt", 1)
                ),
                "gcs_in_backlog": g.get("gcs_in_backlog", 0),
                "exec_queue": g.get("exec_queue", 0),
                "link_backlog_total": g.get("link_backlog_total", 0),
                "link_backlog_max": g.get("link_backlog_max", 0),
                "nodes_w1_converged": g.get("nodes_w1_converged", 0),
                "gcs_busy_s": b.get("gcs_busy_s", 0.0),
            }
            tb = g.get("trace_backlog")
            if tb:
                for i, v in enumerate(tb):
                    rec[f"trace_link_backlog_{i}"] = v
            rows.append(rec)
        return rows

    def write_per_second_csv(self, path: str, num_nodes: int) -> None:
        rows = self.per_second_records(num_nodes)
        if not rows:
            return
        fieldnames: List[str] = list(rows[0].keys())
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def write_detail_csv(self, path: str, limit: Optional[int] = None) -> None:
        rows = self.detail if limit is None else self.detail[:limit]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(DETAIL_FIELDS)
            writer.writerows(rows)

    # ------------------------------------------------------------------ #
    # Aggregate summaries
    # ------------------------------------------------------------------ #
    def wave_summary(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for wave, label in ((1, "wave1"), (2, "wave2")):
            a = self.agg[wave]
            detail_wait = [r[13] for r in self.detail if r[3] == wave]
            detail_e2e = [r[14] for r in self.detail if r[3] == wave]
            detail_gcsq = [r[12] for r in self.detail if r[3] == wave and not math.isnan(r[12])]
            detail_backlog = [r[10] for r in self.detail if r[3] == wave]
            cnt = a.get("wait_cnt", 0.0) or 1.0
            ecnt = a.get("e2e_cnt", 0.0) or 1.0
            out[label] = {
                "messages_delivered": float(a.get("wait_cnt", 0.0)),
                "mean_link_queue_wait_s": a.get("wait_sum", 0.0) / cnt,
                "p50_link_queue_wait_s": _pct(detail_wait, 50),
                "p95_link_queue_wait_s": _pct(detail_wait, 95),
                "mean_end_to_end_s": a.get("e2e_sum", 0.0) / ecnt,
                "p50_end_to_end_s": _pct(detail_e2e, 50),
                "p95_end_to_end_s": _pct(detail_e2e, 95),
                "max_end_to_end_s": a.get("e2e_max", 0.0),
                "mean_gcs_queue_delay_s": (
                    a.get("gcsq_sum", 0.0) / (a.get("gcsq_cnt", 0.0) or 1.0)
                ),
                "max_gcs_queue_delay_s": a.get("gcsq_max", 0.0),
                "mean_backlog_at_enqueue": (
                    a.get("backlog_sum", 0.0) / (a.get("backlog_cnt", 0.0) or 1.0)
                ),
                "max_backlog_at_enqueue": a.get("backlog_max", 0.0),
                "p95_gcs_queue_delay_s": _pct(detail_gcsq, 95),
                "p95_backlog_at_enqueue": _pct(detail_backlog, 95),
            }
        return out

