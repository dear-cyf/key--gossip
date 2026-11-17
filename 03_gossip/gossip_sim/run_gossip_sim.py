#!/usr/bin/env python3
"""Gossip vs star (and star+batching) propagation experiment for RaySyncer.

Control variables (all taken from the existing simulations, unmodified):

* resource-update generator: wave 1 (one update per Raylet) + the milestone
  driven wave 2 feedback, same seeds and same RNG consumption order;
* network/executor model: ``uplink_us`` / ``prop_us`` / ``gcs_fixed_us`` /
  ``gcs_fanout_us`` / ``gcs_write_us`` + per-link jitter;
* de-duplication and acceptance rule: ``PushToSendingQueue`` /
  ``ConsumeSyncMessage`` version semantics;
* payload size: analytic ``RaySyncMessage`` size (``params.estimate_message_bytes``).

The only independent variable is the propagation protocol:
``star`` (stage 1), ``star+batching`` (stage 2), ``gossip`` (this work).

Examples
--------
    python run_gossip_sim.py --nodes 100 300 --arms star_pre star_batch500 gossip
    python run_gossip_sim.py --nodes 1000 --arms star_pre star_batch500 gossip
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from typing import Any, Dict, List

from gossip_sim import (
    GossipParams,
    MetricRecorder,
    estimate_message_bytes,
    stage1_config_class,
    stage2_config_class,
)
from gossip_sim.core import GossipSimulator
from gossip_sim.metrics import metrics_for_gossip, metrics_for_star_like

HERE = os.path.dirname(os.path.abspath(__file__))

SEED = 20251011


# --------------------------------------------------------------------------- #
def build_arm(arm: str, n: int, seed: int = SEED):
    """Return (cfg, kind) with all shared parameters at their stage-1/2 defaults."""
    S1 = stage1_config_class()
    S2 = stage2_config_class()
    if arm == "star_pre":
        return S1(num_nodes=n, seed=seed), "star"
    if arm == "star_default":
        # post-PR merged default: size=1 / delay=0 (still drains the buffer)
        return (S2(num_nodes=n, seed=seed, post_pr_send_path=True,
                   max_batch_size=1, max_batch_delay_ms=0.0), "star")
    if arm == "star_batch100":
        return (S2(num_nodes=n, seed=seed, post_pr_send_path=True,
                   max_batch_size=1000, max_batch_delay_ms=100.0), "star")
    if arm == "star_batch500":
        return (S2(num_nodes=n, seed=seed, post_pr_send_path=True,
                   max_batch_size=1000, max_batch_delay_ms=500.0), "star")
    if arm.startswith("gossip"):
        cfg = S1(num_nodes=n, seed=seed)
        return cfg, "gossip"
    raise SystemExit(f"unknown arm: {arm}")


def run_star_arm(arm: str, n: int, msg_bytes: int, frame_overhead: int,
                 quiet: bool = False, seed: int = SEED,
                 max_sim_s: float | None = None):
    cfg, _ = build_arm(arm, n, seed)
    if max_sim_s is not None:
        cfg.max_sim_s = max_sim_s
    rec = MetricRecorder(n)
    if arm == "star_pre":
        from raysyncer_sim import Simulator
    else:
        from batch_sim import Simulator
    sim = Simulator(cfg, rec)
    t0 = time.perf_counter()
    res = sim.run()
    wall = time.perf_counter() - t0
    row = metrics_for_star_like(arm, n, res, rec, msg_bytes, frame_overhead,
                                batch_delay_ms=getattr(cfg, "max_batch_delay_ms", 0.0),
                                batch_size=getattr(cfg, "max_batch_size", 1))
    row["wall_clock_s"] = round(wall, 2)
    row["wave2_first_emit_s"] = rec.first_wave2_emit_t
    if not quiet:
        print(f"  {arm:14s} N={n:5d}  writes={row['propagation_writes']:>10,}  "
              f"msgs={row['propagation_msgs']:>10,}  "
              f"lat_p95={row['lat_p95_ms']:>9.2f} ms  "
              f"conv={row['convergence_final_s']:>8.2f} s  ({wall:.1f}s)")
    return row, res, rec


def run_gossip_arm(arm: str, n: int, msg_bytes: int, frame_overhead: int,
                   k: int, f: int, exclude_last_hop: bool,
                   ae_period_ms: float = 0.0, ae_peers: int = 1,
                   ae_mode: str = "push",
                   fanout_mode: str = "hash", neighbor_mode: str = "rendezvous",
                   quiet: bool = False, seed: int = SEED,
                   max_sim_s: float | None = None,
                   ae_max_sim_s: float | None = None,
                   ae_deduplicate_inflight: bool = True):
    cfg, _ = build_arm("gossip", n, seed)
    if max_sim_s is not None:
        cfg.max_sim_s = max_sim_s
    gp = GossipParams(degree_k=k, fanout_f=f,
                      exclude_last_hop=exclude_last_hop,
                      antientropy_period_ms=ae_period_ms,
                      antientropy_peers=ae_peers,
                      ae_mode=ae_mode,
                      fanout_mode=fanout_mode, neighbor_mode=neighbor_mode,
                      ae_max_sim_s=ae_max_sim_s,
                      ae_deduplicate_inflight=ae_deduplicate_inflight)
    rec = MetricRecorder(n)
    sim = GossipSimulator(cfg, gp, rec)
    t0 = time.perf_counter()
    res = sim.run()
    wall = time.perf_counter() - t0
    row = metrics_for_gossip(arm, n, res, rec, msg_bytes, frame_overhead)
    row["wall_clock_s"] = round(wall, 2)
    row["wave2_first_emit_s"] = res.wave2_first_emit_t
    if not quiet:
        print(f"  {arm:14s} N={n:5d}  writes={row['propagation_writes']:>10,}  "
              f"msgs={row['propagation_msgs']:>10,}  "
              f"lat_p95={row['lat_p95_ms']:>9.2f} ms  "
              f"conv={row['convergence_final_s']:>8.2f} s  "
              f"stale={100.0 * row['stale_ratio']:.2f}%  "
              f"cov={100.0 * row['final_version_coverage']:.1f}%  ({wall:.1f}s)")
    return row, res, rec


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes", type=int, nargs="+", default=[100, 300])
    ap.add_argument("--arms", nargs="+",
                    default=["star_pre", "star_batch100", "star_batch500", "gossip"])
    ap.add_argument("--gossip-k", type=int, default=4)
    ap.add_argument("--gossip-f", type=int, default=2)
    ap.add_argument("--no-exclude-last-hop", action="store_true",
                    help="match current C++ (only the origin is filtered)")
    ap.add_argument("--ae-period-ms", type=float, default=100.0,
                    help="digest round period for the gossip_ae arm")
    ap.add_argument("--ae-peers", type=int, default=None,
                    help="neighbours per digest round (default: = gossip k)")
    ap.add_argument("--max-sim-s", type=float, default=20000.0)
    ap.add_argument("--ae-max-sim-s", type=float, default=None,
                    help="optional legacy AE cutoff; default uses --max-sim-s")
    ap.add_argument("--no-ae-inflight-dedup", action="store_true")
    ap.add_argument("--ae-mode", choices=["push", "pushpull"], default="push",
                    help="anti-entropy exchange style (default push)")
    ap.add_argument("--fanout-mode", choices=["hash", "random"], default="hash",
                    help="per-message target selection (default hash)")
    ap.add_argument("--neighbor-mode", choices=["rendezvous", "random"],
                    default="rendezvous",
                    help="out-neighbour construction (default rendezvous)")
    ap.add_argument("--arm-suffix", default=None,
                    help="override the arm label suffix (default: derived)")
    ap.add_argument("--msg-bytes", type=int, default=None)
    ap.add_argument("--frame-overhead", type=int, default=64)
    ap.add_argument("--outdir", default=os.path.join(HERE, "results"))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    sizes = estimate_message_bytes()
    msg_bytes = int(args.msg_bytes or sizes["total"])

    os.makedirs(args.outdir, exist_ok=True)
    per_run_dir = os.path.join(args.outdir, "per_run")
    os.makedirs(per_run_dir, exist_ok=True)

    print(f"[setup] seed={args.seed}  msg_bytes={msg_bytes}  "
          f"frame_overhead={args.frame_overhead} B/write")
    print(f"[setup] payload breakdown: {sizes}")
    print(f"[setup] gossip k={args.gossip_k} f={args.gossip_f} "
          f"exclude_last_hop={not args.no_exclude_last_hop}")

    rows: List[Dict[str, Any]] = []
    for n in args.nodes:
        print(f"[run] N={n}")
        for arm in args.arms:
            if arm.startswith("gossip"):
                ae_on = arm.endswith("_ae")
                ae = args.ae_period_ms if ae_on else 0.0
                ae_mode = args.ae_mode if ae_on else "push"
                peers = (args.ae_peers if args.ae_peers is not None
                         else args.gossip_k) if ae_on else 1
                suffix = args.arm_suffix
                if suffix is None:
                    suffix = ""
                    if args.fanout_mode != "hash":
                        suffix += "_frand"
                    if args.neighbor_mode != "rendezvous":
                        suffix += "_nrand"
                    if ae_on and args.ae_mode == "pushpull":
                        suffix += "_pp"
                label = arm + suffix
                row, res, rec = run_gossip_arm(
                    label, n, msg_bytes, args.frame_overhead,
                    args.gossip_k, args.gossip_f,
                    exclude_last_hop=not args.no_exclude_last_hop,
                    ae_period_ms=ae, ae_peers=peers,
                    ae_mode=ae_mode,
                    fanout_mode=args.fanout_mode,
                    neighbor_mode=args.neighbor_mode,
                    seed=args.seed, max_sim_s=args.max_sim_s,
                    ae_max_sim_s=args.ae_max_sim_s,
                    ae_deduplicate_inflight=not args.no_ae_inflight_dedup)
                extra = {
                    "topology": res.topology,
                    "per_node_sent": res.per_node_sent.tolist(),
                    "pushes_attempted": int(res.pushes_attempted),
                    "pushes_suppressed_peer_has_it": int(res.pushes_suppressed_peer_has_it),
                    "pushes_excluded_no_target": int(res.pushes_excluded),
                    "uplink_stale": int(res.uplink_stale),
                    "stale_received": int(res.stale_received),
                    "ae_data_msgs": int(res.ae_data_msgs),
                    "ae_digest_bytes": int(res.ae_digest_bytes),
                    "ae_stopped_early": int(res.ae_stopped_early),
                    "coverage_trace": res.coverage_trace,
                    "stale_repairs": int(res.stale_repairs),
                }
            else:
                row, res, rec = run_star_arm(arm, n, msg_bytes,
                                             args.frame_overhead,
                                             seed=args.seed,
                                             max_sim_s=args.max_sim_s)
                extra = {
                    "batch_sizes": {str(kk): int(vv) for kk, vv in
                                    sorted(rec.batch_sizes.items())[:12]},
                    "batch_size_max": int(rec.max_batch or 1),
                }
            row["seed"] = args.seed          # record it so a run is traceable
            rows.append(row)
            tag = f"{row['arm']}_N{n}"
            with open(os.path.join(per_run_dir, f"{tag}.json"), "w") as fh:
                json.dump({"metrics": row, "extra": extra,
                           "msg_bytes_breakdown": sizes}, fh, indent=1)

    # ---- summary.csv ------------------------------------------------------ #
    cols: List[str] = []
    for r in rows:
        for k_ in r:
            if k_ not in cols:
                cols.append(k_)
    csv_path = os.path.join(args.outdir, "summary.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(args.outdir, "summary.json"), "w") as fh:
        json.dump({"rows": rows, "msg_bytes": msg_bytes,
                   "msg_bytes_breakdown": sizes, "seed": args.seed}, fh, indent=1)
    print(f"[out] {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
