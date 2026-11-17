"""The seven requested indicators, computed identically for every arm.

| indicator                        | definition                                                    |
|----------------------------------|---------------------------------------------------------------|
| mean / p95 / p99 propagation     | per delivered update: ``t_delivery - t_emit`` (all deliveries)|
| convergence time                 | wave 1 and final: ``max over (src,dst) of (first/last) recv``  |
| total messages                   | uplink + propagation (downlink writes x msgs, or relays)       |
| total transmitted bytes          | messages x payload size, plus per-write frame overhead         |
| max single-node send volume      | propagation messages sent by the busiest node (star: the GCS)  |
| stale-state ratio                | stale/duplicate drops / all received messages                  |

The GCS is a *node* for the "busiest sender" metric on purpose: that is exactly
the single point the gossip overlay is supposed to remove.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Optional

import numpy as np


def pct(values, q: float) -> float:
    """Percentile definition identical to stage-1's monitor ``_pct``."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def _base_row(arm: str, n: int, extras: Optional[Dict[str, Any]] = None):
    row: "OrderedDict[str, Any]" = OrderedDict()
    row["arm"] = arm
    row["num_nodes"] = n
    if extras:
        row.update(extras)
    return row


def metrics_for_star_like(arm: str, n: int, res, rec, msg_bytes: int,
                          frame_overhead: int, batch_delay_ms: float = 0.0,
                          batch_size: int = 1) -> "OrderedDict[str, Any]":
    """Star arm (stage-1 pre-PR or stage-2 post-PR/batching)."""
    sta = rec.stats(pct)
    prop_msgs = int(sta["messages_written"])
    prop_writes = int(sta["write_events"])
    uplink_msgs = int(res.total_inbound_messages)
    updates = int(res.total_inbound_messages)
    received = uplink_msgs + int(sta["deliveries"])
    stale = int(res.total_stale_dropped)   # raylet-side receives are pre-filtered
    row = _base_row(arm, n, {
        "protocol": "star",
        "batch_size": batch_size,
        "batch_delay_ms": batch_delay_ms,
        "updates_total": updates,
        "updates_wave1": int(res.total_wave1_messages),
        "updates_wave2": int(res.total_wave2_messages),
    })
    row.update(_common_numbers(sta, uplink_msgs, prop_msgs, prop_writes,
                               msg_bytes, frame_overhead))
    row["convergence_wave1_s"] = float(res.wave1_convergence_s)
    row["convergence_final_s"] = float(res.final_convergence_s)
    row["final_version_coverage"] = res.final_version_coverage
    row["final_version_coverage_time_s"] = float(res.final_convergence_s)
    row["uncovered_final_pairs"] = res.uncovered_final_pairs
    row["converged"] = res.converged
    row["termination_reason"] = res.termination_reason
    row["max_node_sends"] = prop_msgs
    row["max_node_sends_role"] = "GCS"
    row["max_node_sends_per_update"] = prop_msgs / max(1, updates)
    row["received_msgs"] = received
    row["stale_msgs"] = stale
    row["stale_ratio"] = stale / received if received else 0.0
    row["relay_msgs_per_update"] = prop_msgs / max(1, updates)
    row["write_events_per_update"] = prop_writes / max(1, updates)
    row["wall_clock_s"] = None
    return row


def metrics_for_gossip(arm: str, n: int, res, rec, msg_bytes: int,
                       frame_overhead: int) -> "OrderedDict[str, Any]":
    sta = rec.stats(pct)
    uplink_msgs = int(res.uplink_msgs)
    prop_msgs = int(res.relay_msgs)
    prop_writes = int(res.relay_writes)
    updates = uplink_msgs
    received = uplink_msgs + int(res.received_total)
    stale = int(res.uplink_stale + res.stale_received)
    gp = res.gparams
    row = _base_row(arm, n, {
        "protocol": "gossip",
        "batch_size": 0,
        "batch_delay_ms": 0.0,
        "degree_k": gp.degree_k,
        "fanout_f": gp.fanout_f,
        "exclude_last_hop": int(gp.exclude_last_hop),
        "topology_undirected_diameter": res.topology["undirected_diameter"],
        "topology_directed_edges": res.topology["directed_edges"],
        "updates_total": updates,
        "updates_wave1": int(res.total_wave1_updates),
        "updates_wave2": int(res.total_wave2_updates),
    })
    row.update(_common_numbers(sta, uplink_msgs, prop_msgs, prop_writes,
                               msg_bytes, frame_overhead))
    row["convergence_wave1_s"] = float(res.wave1_convergence_s)
    row["convergence_final_s"] = float(res.final_convergence_s)
    row["final_version_coverage"] = float(res.final_version_coverage)
    row["final_version_coverage_time_s"] = float(res.final_version_coverage_time_s)
    row["uncovered_final_pairs"] = int(res.uncovered_final_pairs)
    row["max_node_sends"] = int(res.max_node_sends)
    row["max_node_sends_role"] = f"raylet#{res.max_node_sends_node}"
    row["max_node_sends_per_update"] = res.max_node_sends / max(1, updates)
    row["received_msgs"] = received
    row["stale_msgs"] = stale
    row["stale_ratio"] = stale / received if received else 0.0
    row["relay_msgs_per_update"] = prop_msgs / max(1, updates)
    row["write_events_per_update"] = prop_writes / max(1, updates)
    row["pushes_attempted"] = int(res.pushes_attempted)
    row["pushes_suppressed_peer_has_it"] = int(res.pushes_suppressed_peer_has_it)
    row["pushes_excluded_no_target"] = int(res.pushes_excluded)
    # anti-entropy (0 when the arm runs pure push gossip)
    ae_data = int(res.ae_data_sent)
    ae_bytes = int(res.ae_digest_bytes) + ae_data * msg_bytes
    row["ae_enabled"] = int(gp.antientropy_period_ms > 0)
    row["ae_period_ms"] = float(gp.antientropy_period_ms)
    row["ae_slices"] = int(gp.antientropy_slices)
    row["ae_rounds"] = int(res.ae_rounds)
    row["ae_data_msgs"] = ae_data
    row["ae_data_enqueued"] = int(res.ae_data_msgs)
    row["ae_repairs_suppressed"] = int(res.ae_repairs_suppressed)
    row["ae_pending_peak"] = int(res.ae_pending_peak)
    row["ae_pending_at_end"] = int(res.ae_pending_at_end)
    row["ae_repair_overwrites"] = int(res.ae_repair_overwrites)
    row["ae_requests_suppressed"] = int(res.ae_requests_suppressed)
    row["ae_requests_pruned"] = int(res.ae_requests_pruned)
    row["ae_digest_overwrites"] = int(res.ae_digest_overwrites)
    row["ae_deduplicate_inflight"] = gp.ae_deduplicate_inflight
    row["ae_deadline_s"] = gp.resolved_ae_deadline_s(res.cfg)
    row["termination_reason"] = res.termination_reason
    row["ae_push_msgs"] = int(res.ae_push_msgs)
    row["ae_pull_msgs"] = int(res.ae_pull_msgs)
    row["ae_mode"] = gp.ae_mode
    row["ae_digest_msgs"] = int(res.ae_digest_msgs)
    row["ae_digest_bytes"] = int(res.ae_digest_bytes)
    row["ae_stopped_early"] = int(res.ae_stopped_early)
    row["ae_finished_s"] = (float(res.ae_all_milestones_reached_t)
                            if res.ae_all_milestones_reached_t >= 0 else None)
    row["total_data_messages"] = int(row["total_messages"]) + ae_data
    row["total_messages"] = row["total_data_messages"] + int(res.ae_digest_msgs)
    row["total_bytes_payload"] = int(row["total_bytes_payload"]) + ae_bytes
    row["total_bytes_wire_est"] = (int(row["total_bytes_wire_est"]) + ae_bytes
                                   + (ae_data + int(res.ae_digest_msgs)) * frame_overhead)
    row["wall_clock_s"] = None
    return row


def _common_numbers(sta, uplink_msgs: int, prop_msgs: int, prop_writes: int,
                    msg_bytes: int, frame_overhead: int) -> Dict[str, Any]:
    payload_bytes = (uplink_msgs + prop_msgs) * msg_bytes
    wire_bytes = payload_bytes + (uplink_msgs + prop_writes) * frame_overhead
    return {
        # --- latency (every accepted delivery) ---
        "lat_mean_ms": sta["lat_all_mean_ms"],
        "lat_p50_ms": sta["lat_all_p50_ms"],
        "lat_p95_ms": sta["lat_all_p95_ms"],
        "lat_p99_ms": sta["lat_all_p99_ms"],
        "lat_max_ms": sta["lat_all_max_ms"],
        # --- latency, split by wave (same split as stage 1) ---
        "lat_w1_mean_ms": sta["lat_w1_mean_ms"],
        "lat_w1_p95_ms": sta["lat_w1_p95_ms"],
        "lat_w2_mean_ms": sta["lat_w2_mean_ms"],
        "lat_w2_p95_ms": sta["lat_w2_p95_ms"],
        # --- latency of the first delivery per (src, dst) pair ---
        "lat_first_pair_mean_ms": sta["lat_first_pair_mean_ms"],
        "lat_first_pair_p95_ms": sta["lat_first_pair_p95_ms"],
        "lat_first_pair_p99_ms": sta["lat_first_pair_p99_ms"],
        # --- volume ---
        "uplink_msgs": uplink_msgs,
        "propagation_msgs": prop_msgs,
        "total_messages": uplink_msgs + prop_msgs,
        "propagation_writes": prop_writes,
        "msg_bytes": msg_bytes,
        "total_bytes_payload": payload_bytes,
        "total_bytes_wire_est": wire_bytes,
        "deliveries": int(sta["deliveries"]),
    }
