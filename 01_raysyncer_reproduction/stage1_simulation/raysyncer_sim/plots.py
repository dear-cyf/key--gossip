"""Plot generation for the Ray Syncer reproduction (matplotlib, headless)."""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .core import SimResult
from .monitor import Recorder

C_EMIT = "#1f77b4"
C_RECV = "#d62728"
C_BCAST = "#2ca02c"
C_WRITE = "#ff7f0e"
C_QUEUE = "#9467bd"
C_W2 = "#8c564b"


def _save(fig, path: str) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=135)
    plt.close(fig)
    return path


def _series(rows: Sequence[Dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(r.get(key, 0.0) or 0.0) for r in rows], dtype=np.float64)


def _mark_waves(ax, wave2_start: Optional[float], wave1_conv: Optional[float]) -> None:
    if wave1_conv:
        ax.axvline(wave1_conv, color="#999999", ls=":", lw=1.2)
    if wave2_start:
        ax.axvline(wave2_start, color=C_W2, ls="--", lw=1.3)


# ---------------------------------------------------------------------- #
# Per-run figures
# ---------------------------------------------------------------------- #
def plot_throughput(rows, meta, out) -> str:
    t = _series(rows, "second")
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax = axes[0]
    ax.plot(t, _series(rows, "updates_emitted"), color=C_EMIT, lw=1.6,
            label="Raylet updates produced")
    ax.plot(t, _series(rows, "gcs_received"), color=C_RECV, lw=1.8,
            label="GCS received (inbound sync msgs)")
    ax.plot(t, _series(rows, "gcs_broadcasts"), color=C_BCAST, lw=1.8,
            label="GCS broadcasts (fan-out ops)")
    ax.plot(t, _series(rows, "link_writes"), color=C_WRITE, lw=1.4, alpha=0.9,
            label="GCS -> Raylet writes completed")
    ax.set_ylabel("messages / second")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_title(
        f"GCS receive vs. broadcast rate  (N={meta['num_nodes']} nodes, "
        f"no batching)"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25, which="both")
    _mark_waves(ax, meta.get("wave2_start_s"), meta.get("wave1_convergence_s"))

    ax = axes[1]
    ax.plot(t, _series(rows, "avg_node_sent_per_sec"), color=C_EMIT, lw=1.6,
            label="avg node SENT / s")
    ax.plot(t, _series(rows, "avg_node_recv_per_sec"), color=C_WRITE, lw=1.6,
            label="avg node RECEIVED / s")
    ax.set_xlabel("simulated time (s)")
    ax.set_ylabel("messages / node / s")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    _mark_waves(ax, meta.get("wave2_start_s"), meta.get("wave1_convergence_s"))
    return _save(fig, out)


def plot_queues(rows, meta, out, control_rows=None) -> str:
    t = _series(rows, "second")
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax = axes[0]
    ax.plot(t, _series(rows, "link_backlog_total"), color=C_QUEUE, lw=1.8,
            label="total per-connection backlog (sum of sending_buffer_)")
    ax.plot(t, _series(rows, "link_backlog_max"), color="#e377c2", lw=1.5,
            label="deepest single connection backlog")
    ax.plot(t, _series(rows, "gcs_in_backlog"), color=C_RECV, lw=1.6,
            label="GCS inbound backlog (received, not yet fanned out)")
    if control_rows:
        ax.plot(_series(control_rows, "second"),
                _series(control_rows, "link_backlog_total"),
                color="#9e9e9e", ls="--", lw=1.6,
                label="wave-1 only, control (backlog drains freely)")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylabel("queue length (log scale)")
    ax.set_title(f"Queue build-up in the sync path  (N={meta['num_nodes']} nodes)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25, which="both")
    _mark_waves(ax, meta.get("wave2_start_s"), meta.get("wave1_convergence_s"))

    ax = axes[1]
    trace_keys: List[str] = []
    for row in rows:
        for k in row:
            if k.startswith("trace_link_backlog_") and k not in trace_keys:
                trace_keys.append(k)
    trace_keys.sort(key=lambda k: int(k.rsplit("_", 1)[1]))
    for i, k in enumerate(trace_keys[:4]):
        ax.plot(t, _series(rows, k), lw=1.4, label=f"connection backlog (dst node sample {i})")
    ax.set_xlabel("simulated time (s)")
    ax.set_ylabel("messages in sending_buffer_")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    _mark_waves(ax, meta.get("wave2_start_s"), meta.get("wave1_convergence_s"))
    return _save(fig, out)


def plot_latency(rows, meta, out) -> str:
    t = _series(rows, "second")
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(t, _series(rows, "propagation_latency_avg_ms"), color=C_RECV, lw=1.8,
            label="propagation latency: update produced -> applied (mean)")
    ax.plot(t, _series(rows, "link_queue_wait_avg_ms"), color=C_QUEUE, lw=1.6,
            label="time waiting in the connection backlog (mean)")
    ax.plot(t, _series(rows, "gcs_queue_delay_avg_ms"), color=C_BCAST, lw=1.6,
            label="time waiting for GCS fan-out (mean)")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("simulated time (s)")
    ax.set_ylabel("latency (ms, log scale)")
    ax.set_title(f"Propagation latency over time  (N={meta['num_nodes']} nodes)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25, which="both")
    _mark_waves(ax, meta.get("wave2_start_s"), meta.get("wave1_convergence_s"))
    w2 = meta.get("wave2_start_s") or 0.0
    ax.annotate("wave 2 enters the backlog\n(updates keep queueing behind it)",
                xy=(w2, 0.30), xycoords=("data", "axes fraction"),
                xytext=(0.12, 0.90), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=C_W2), color=C_W2, fontsize=8)
    return _save(fig, out)


def plot_convergence(rows, res: SimResult, meta, out, control_rows=None) -> str:
    t = _series(rows, "second")
    n = meta["num_nodes"]
    ideal = float(n * (n - 1))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True)

    ax = axes[0]
    frac_w1 = _series(rows, "nodes_w1_converged") / max(1, n)
    ax.step(t, 100.0 * frac_w1, where="post", color=C_EMIT, lw=1.8,
            label="Raylets that applied every wave-1 update")

    last = np.asarray(res.last_delivery_per_dst, dtype=np.float64)
    last = last[last >= 0]
    if last.size:
        last.sort()
        cdf = 100.0 * np.arange(1, last.size + 1) / last.size
        ax.step(np.concatenate([[0.0], last]), np.concatenate([[0.0], cdf]),
                where="post", color=C_W2, lw=1.8,
                label="Raylets that applied every final version")

    if control_rows:
        cf = _series(control_rows, "nodes_w1_converged") / max(1, n)
        ax.step(_series(control_rows, "second"), 100.0 * cf, where="post",
                color="#9e9e9e", ls="--", lw=1.6,
                label="control (wave 1 only)")

    ax.axvline(res.wave1_convergence_s, color="#999999", ls=":", lw=1.2)
    ax.axvline(res.final_convergence_s, color=C_W2, ls="--", lw=1.3)
    extra = res.final_convergence_s - res.wave1_convergence_s
    ax.annotate("", xy=(res.final_convergence_s, 55), xytext=(res.wave1_convergence_s, 55),
                arrowprops=dict(arrowstyle="<->", color=C_W2))
    ax.text((res.wave1_convergence_s + res.final_convergence_s) / 2.0, 58,
            f"wave-2 adds {extra:.0f} s", ha="center", color=C_W2, fontsize=9)
    ax.set_ylabel("% of Raylets converged")
    ax.set_ylim(-4, 112)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    wave1_label = (f"{res.wave1_convergence_s:.1f} s"
                   if np.isfinite(res.wave1_convergence_s) else "not reached")
    final_label = (f"{res.final_convergence_s:.1f} s"
                   if np.isfinite(res.final_convergence_s) else "not reached")
    ax.set_title(f"Convergence and applied-work  (N={meta['num_nodes']} nodes): "
                 f"wave 1 {wave1_label} -> final {final_label}")

    ax = axes[1]
    cum = np.cumsum(_series(rows, "link_writes")) / max(1, n)
    ax.plot(t, cum, color=C_WRITE, lw=1.9,
            label="messages applied per Raylet (wave 1 + wave 2)")
    if control_rows:
        c2 = np.cumsum(_series(control_rows, "link_writes")) / max(1, n)
        ax.plot(_series(control_rows, "second"), c2, color="#9e9e9e", ls="--",
                lw=1.6, label="control (wave 1 only)")
    ax.axhline(n - 1, color=C_EMIT, ls=":", lw=1.2)
    ax.text(t[0] + 0.2, n - 1, " N-1 = one full cluster-view refresh",
            color=C_EMIT, fontsize=8, va="bottom")
    ax.set_xlabel("simulated time (s)")
    ax.set_ylabel("cumulative messages\napplied per Raylet")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    for a in axes:
        if np.isfinite(res.wave1_convergence_s):
            a.axvline(res.wave1_convergence_s, color="#999999", ls=":", lw=1.2)
        if np.isfinite(res.final_convergence_s):
            a.axvline(res.final_convergence_s, color=C_W2, ls="--", lw=1.3)
    return _save(fig, out)


def plot_wave_comparison(wave_summary, meta, out) -> str:
    labels = ["mean backlog\nwhen enqueued", "p95 backlog\nwhen enqueued",
              "mean backlog wait\n(s)", "p95 backlog wait\n(s)",
              "mean propagation\nlatency (s)", "p95 propagation\nlatency (s)"]
    w1 = wave_summary["wave1"]
    w2 = wave_summary["wave2"]
    v1 = [w1["mean_backlog_at_enqueue"], w1["p95_backlog_at_enqueue"],
          w1["mean_link_queue_wait_s"], w1["p95_link_queue_wait_s"],
          w1["mean_end_to_end_s"], w1["p95_end_to_end_s"]]
    v2 = [w2["mean_backlog_at_enqueue"], w2["p95_backlog_at_enqueue"],
          w2["mean_link_queue_wait_s"], w2["p95_link_queue_wait_s"],
          w2["mean_end_to_end_s"], w2["p95_end_to_end_s"]]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(x - 0.2, v1, 0.4, color=C_EMIT, label="wave 1 (PG creation burst)")
    ax.bar(x + 0.2, v2, 0.4, color=C_W2, label="wave 2 (post-scheduling updates)")
    for xi, (a, b) in enumerate(zip(v1, v2)):
        ax.text(xi - 0.2, a, f"{a:.2f}", ha="center", va="bottom", fontsize=7)
        ax.text(xi + 0.2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_title("Wave 2 enters an already deep backlog "
                 f"(N={meta['num_nodes']} nodes)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y", which="both")
    return _save(fig, out)


# ---------------------------------------------------------------------- #
# Scaling figures
# ---------------------------------------------------------------------- #
def plot_scaling_messages(scaling: List[Dict[str, Any]], out: str) -> str:
    n = np.asarray([s["num_nodes"] for s in scaling], dtype=np.float64)
    writes = np.asarray([s["total_link_writes"] for s in scaling], dtype=np.float64)
    ideal = n * (n - 1.0)
    order = np.argsort(n)
    quad = np.polyfit(n[order], writes[order], 2)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(n, ideal, "o--", color="#bbbbbb", label="N x (N-1) ideal wave-1 fan-out")
    ax.plot(n, writes, "o-", color=C_WRITE, lw=1.8,
            label="simulated messages written (wave 1 + wave 2)")
    xs = np.linspace(n.min(), n.max(), 100)
    ax.plot(xs, np.polyval(quad, xs), ":", color=C_W2,
            label=f"quadratic fit {quad[0]:.3f} N^2")
    ax.set_xlabel("cluster size N (nodes)")
    ax.set_ylabel("messages written by the GCS")
    ax.set_title("O(N^2) sync amplification: one update per node -> N-1 broadcasts")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, out)


def plot_scaling_times(scaling: List[Dict[str, Any]], out: str) -> str:
    n = np.asarray([s["num_nodes"] for s in scaling], dtype=np.float64)
    w1 = np.asarray([s["wave1_convergence_s"] for s in scaling])
    fin = np.asarray([s["final_convergence_s"] for s in scaling])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(n, w1, "o-", color=C_EMIT, lw=1.8, label="wave-1 convergence")
    ax.plot(n, fin, "s-", color=C_W2, lw=1.8, label="final convergence (wave 1+2)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cluster size N (nodes, log)")
    ax.set_ylabel("time to converge (s, log)")
    ax.set_title("Convergence time grows super-linearly with cluster size")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")
    return _save(fig, out)


def plot_loglog(scaling: List[Dict[str, Any]], out: str) -> str:
    n = np.asarray([s["num_nodes"] for s in scaling], dtype=np.float64)
    writes = np.asarray([s["total_link_writes"] for s in scaling], dtype=np.float64)
    slope, intercept = np.polyfit(np.log(n), np.log(writes), 1)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(np.log(n), np.log(writes), "o", color=C_WRITE, label="simulation")
    xs = np.linspace(np.log(n.min()), np.log(n.max()), 50)
    ax.plot(xs, slope * xs + intercept, "--", color="#333333",
            label=f"fitted exponent = {slope:.2f}")
    ax.set_xlabel("ln(N)")
    ax.set_ylabel("ln(messages written)")
    ax.set_title("Log-log scaling: fitted exponent ~2 confirms the O(N^2) blow-up")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, out)


def render_all(res: SimResult, rows, wave_summary, meta, outdir: str,
               control_rows=None) -> List[str]:
    paths = [
        plot_throughput(rows, meta, os.path.join(outdir, "01_gcs_receive_vs_broadcast.png")),
        plot_queues(rows, meta, os.path.join(outdir, "02_queue_length.png"), control_rows),
        plot_latency(rows, meta, os.path.join(outdir, "03_propagation_latency.png")),
        plot_convergence(rows, res, meta, os.path.join(outdir, "04_convergence.png"), control_rows),
        plot_wave_comparison(wave_summary, meta, os.path.join(outdir, "05_wave12_comparison.png")),
        plot_write_breakdown(res, os.path.join(outdir, "09_write_breakdown.png")),
    ]
    return paths


def plot_write_breakdown(res: SimResult, out: str) -> str:
    n = res.num_nodes
    merged = float(res.total_inplace_merges)
    new = float(res.total_new_entry_writes)
    slot = float(res.total_slot_writes)
    resend = float(res.total_resend_writes)
    pushes = merged + new

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    ax = axes[0]
    ax.barh([0], [merged / 1000.0], color="#aec7e8",
            label=f"merged into a slot still queued ({merged:,.0f})")
    ax.barh([0], [new / 1000.0], left=[merged / 1000.0], color=C_EMIT,
            label=f"created a new entry = 1 write ({new:,.0f})")
    ax.set_yticks([0])
    ax.set_yticklabels([f"PushToSendingQueue\ncalls: {pushes:,.0f}"], fontsize=8)
    ax.set_xlabel("calls (thousands)")
    ax.set_title("Fan-out push results", fontsize=10)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.text(merged / 2000.0, 0, f"{merged/pushes*100:.1f}%\nfree", ha="center",
            va="center", fontsize=8, color="#12395c")

    ax = axes[1]
    ax.barh([0], [slot / 1000.0], color=C_BCAST,
            label=f"first-time slot, N(N-1) ({slot:,.0f})")
    ax.barh([0], [resend / 1000.0], left=[slot / 1000.0], color=C_W2,
            label=f"wave-2 re-sends (extra) ({resend:,.0f})")
    ax.set_yticks([0])
    ax.set_yticklabels([f"GCS writes: {res.total_link_writes:,.0f}"], fontsize=8)
    ax.set_xlabel("messages written (thousands)")
    ax.set_title(f"Where the {res.total_link_writes:,.0f} writes come from", fontsize=10)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.text(slot / 2000.0, 0, f"{slot/res.total_link_writes*100:.1f}%", ha="center",
            va="center", fontsize=8, color="white")
    ax.text((slot + resend / 2.0) / 1000.0, 0, f"{resend/res.total_link_writes*100:.1f}%",
            ha="center", va="center", fontsize=8, color="white")

    ax = axes[2]
    extra_per_conn = res.new_entries_by_src / max(1, n - 1) - 1.0
    ax.bar(np.arange(n), np.clip(extra_per_conn, 0, None), width=1.0, color=C_W2)
    ax.set_xlabel("source node index")
    ax.set_ylabel("extra (re-sent) messages\nper connection")
    ax.set_title("Extra writes contributed by each node", fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    ax.set_ylim(0, max(1.2, float(np.clip(extra_per_conn, 0, None).max()) * 1.2))
    nz = int((np.clip(extra_per_conn, 0, None) > 0).sum())
    ax.text(0.02, 0.88, f"{nz}/{n} nodes contributed 1 re-send;\n"
                        f"{n - nz} contributed none",
            transform=ax.transAxes, fontsize=8, color="#4a2c22")
    return _save(fig, out)


def render_scaling(scaling, outdir: str) -> List[str]:
    return [
        plot_scaling_messages(scaling, os.path.join(outdir, "06_scaling_messages.png")),
        plot_scaling_times(scaling, os.path.join(outdir, "07_scaling_times.png")),
        plot_loglog(scaling, os.path.join(outdir, "08_scaling_loglog.png")),
    ]

