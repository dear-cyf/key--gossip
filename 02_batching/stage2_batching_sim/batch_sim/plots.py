"""Stage-2 figures: pre-batching baseline vs PR #57641 batching.

All figures are written as PNG files into the run's output directory.  Each
figure is self-contained and labelled in Chinese, matching the stage-1 report
style, but the numbers come from the stage-2 comparison runs only.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Chinese labels: pick the first CJK font that exists on this machine.
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
    "WenQuanYi Zen Hei", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

C_BASE = "#c0392b"
C_BATCH = "#1f77b4"


def _series(rows: Sequence[Dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(r.get(key, 0.0) or 0.0) for r in rows])


def _t(rows: Sequence[Dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(r["second"]) for r in rows])


def _finish(fig, path: str, files: List[str]) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    files.append(path)


def fig_write_rate(base_rows, batch_rows, outdir) -> List[str]:
    files: List[str] = []
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    t = _t(base_rows)
    axes[0].plot(t, _series(base_rows, "messages_written"), color=C_BASE,
                 label="无 batching：每条消息一次写")
    tb = _t(batch_rows)
    axes[0].plot(tb, _series(batch_rows, "messages_written"), color=C_BATCH,
                 label="有 batching：实际写出的消息数")
    axes[0].set_ylabel("每秒写出的 RaySyncMessage 数（全集群）")
    axes[0].set_yscale("symlog", linthresh=10)
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, _series(base_rows, "link_writes"), color=C_BASE,
                 label="无 batching：gRPC 写次数")
    axes[1].plot(tb, _series(batch_rows, "link_writes"), color=C_BATCH,
                 label="有 batching：gRPC 写次数（每批一次）")
    axes[1].set_xlabel("仿真时间（秒）")
    axes[1].set_ylabel("每秒 GCS→Raylet 写事件数")
    axes[1].set_yscale("symlog", linthresh=10)
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)
    fig.suptitle("GCS→Raylet 写路径：有/无 batching")
    _finish(fig, os.path.join(outdir, "01_write_rate.png"), files)
    return files


def fig_cumulative(base_rows, batch_rows, outdir, prd_rows=None) -> List[str]:
    files: List[str] = []
    fig, ax = plt.subplots(figsize=(10, 5.5))

    tb = _t(base_rows)
    cb = np.cumsum(_series(base_rows, "link_writes"))
    ax.plot(tb, cb, color=C_BASE, label=f"无 batching（合计 {cb[-1]:,.0f} 次写）")

    ta = _t(batch_rows)
    ca = np.cumsum(_series(batch_rows, "link_writes"))
    ax.plot(ta, ca, color=C_BATCH, label=f"有 batching（合计 {ca[-1]:,.0f} 次写）")

    cm = np.cumsum(_series(batch_rows, "messages_written"))
    ax.plot(ta, cm, color=C_BATCH, ls="--", alpha=0.7,
            label=f"有 batching：携带的消息数（合计 {cm[-1]:,.0f}）")

    if prd_rows:
        tp = _t(prd_rows)
        cp = np.cumsum(_series(prd_rows, "link_writes"))
        ax.plot(tp, cp, color="#2e8b57", ls="-.",
                label=f"PR 合并后默认 1/0ms（合计 {cp[-1]:,.0f} 次写）")

    ax.set_yscale("log")
    ax.set_xlabel("仿真时间（秒）")
    ax.set_ylabel("累计 gRPC 写事件数（log）")
    ax.set_title("累计写出量：batching 把 O(N^2) 次写合并成 ~O(N) 个批")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right")
    _finish(fig, os.path.join(outdir, "02_cumulative_writes.png"), files)
    return files


def fig_convergence(base_rows, batch_rows, outdir) -> List[str]:
    files: List[str] = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    t = _t(base_rows)
    tb = _t(batch_rows)
    axes[0].plot(t, _series(base_rows, "nodes_w1_converged"), color=C_BASE,
                 label="无 batching")
    axes[0].plot(tb, _series(batch_rows, "nodes_w1_converged"), color=C_BATCH,
                 label="有 batching")
    axes[0].set_xlabel("仿真时间（秒）")
    axes[0].set_ylabel("已拿到完整第一波视图的 Raylet 数")
    axes[0].set_title("第一波资源视图收敛")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, _series(base_rows, "link_backlog_total"), color=C_BASE,
                 label="无 batching：全集群排队消息")
    axes[1].plot(tb, _series(batch_rows, "link_backlog_total"), color=C_BATCH,
                 label="有 batching：全集群排队消息")
    axes[1].set_xlabel("仿真时间（秒）")
    axes[1].set_ylabel("sending_buffer_ 中待发消息总数")
    axes[1].set_title("各连接发送缓冲排队深度")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    _finish(fig, os.path.join(outdir, "03_convergence_backlog.png"), files)
    return files


def fig_latency(base_waves, batch_waves, outdir) -> List[str]:
    files: List[str] = []
    waves = [("wave1", "第一波"), ("wave2", "第二波")]
    x = np.arange(len(waves))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, key, title in (
        (axes[0], "mean_end_to_end_s", "平均端到端传播延迟"),
        (axes[1], "p95_end_to_end_s", "P95 端到端传播延迟"),
    ):
        b = [float(base_waves[w].get(key, 0.0) or 0.0) for w, _ in waves]
        a = [float(batch_waves[w].get(key, 0.0) or 0.0) for w, _ in waves]
        ax.bar(x - width / 2, b, width, color=C_BASE, label="无 batching")
        ax.bar(x + width / 2, a, width, color=C_BATCH, label="有 batching")
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in waves])
        ax.set_ylabel("秒")
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        ax.legend()
    _finish(fig, os.path.join(outdir, "04_latency.png"), files)
    return files


def fig_batch_profile(batch_res, batch_rows, outdir) -> List[str]:
    files: List[str] = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    sizes = np.asarray(batch_res.batch_sizes, dtype=np.float64)
    if sizes.size:
        axes[0].hist(sizes, bins=min(40, max(5, int(sizes.max()))), color=C_BATCH,
                     alpha=0.85)
        axes[0].axvline(sizes.mean(), color="k", ls="--",
                        label=f"均值 {sizes.mean():.1f}")
        axes[0].legend()
    axes[0].set_xlabel("单个批携带的消息数")
    axes[0].set_ylabel("批次数")
    axes[0].set_title("批大小分布（有 batching）")
    axes[0].grid(alpha=0.3, axis="y")

    triggers = batch_res.trigger_counts or {}
    labels = {
        "timer": "超时触发",
        "size": "达到 batch size",
        "immediate": "立即发送（无 batching）",
        "write_done": "写完成后补发",
    }
    keys = [k for k in labels if triggers.get(k)]
    if keys:
        vals = [triggers[k] for k in keys]
        axes[1].bar([labels[k] for k in keys], vals, color=C_BATCH, alpha=0.85)
        axes[1].set_ylabel("批次数")
        for i, v in enumerate(vals):
            axes[1].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    axes[1].set_title("批次触发原因")
    axes[1].grid(alpha=0.3, axis="y")
    _finish(fig, os.path.join(outdir, "05_batch_profile.png"), files)
    return files


def fig_sweep(sweep_rows, outdir) -> List[str]:
    """Convergence time vs batch timeout, one line per batch size."""
    files: List[str] = []
    if not sweep_rows:
        return files
    fig, ax = plt.subplots(figsize=(9, 5.2))
    by_size: Dict[int, List[Dict[str, Any]]] = {}
    for r in sweep_rows:
        by_size.setdefault(int(r["batch_size"]), []).append(r)
    for size in sorted(by_size):
        rows = sorted(by_size[size], key=lambda r: r["batch_delay_ms"])
        ax.plot([r["batch_delay_ms"] for r in rows],
                [r["final_convergence_s"] for r in rows],
                marker="o", label=f"batch size = {size}")
    ax.set_xlabel("batch 超时 max_batch_delay_ms（毫秒）")
    ax.set_ylabel("最终收敛时间（秒）")
    ax.set_title("batching 参数扫描")
    ax.grid(alpha=0.3)
    ax.legend()
    _finish(fig, os.path.join(outdir, "06_sweep.png"), files)
    return files


def fig_scaling(scaling_rows, outdir) -> List[str]:
    files: List[str] = []
    if not scaling_rows:
        return files
    n = np.asarray([r["num_nodes"] for r in scaling_rows], dtype=np.float64)
    bw = np.asarray([r["baseline_write_events"] for r in scaling_rows], dtype=np.float64)
    aw = np.asarray([r["batching_write_events"] for r in scaling_rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(n, bw, marker="o", color=C_BASE, label="无 batching")
    axes[0].plot(n, aw, marker="s", color=C_BATCH, label="有 batching")
    axes[0].plot(n, n * (n - 1), ls=":", color="gray", label="N(N-1) 理想扇出")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("节点数 N（log）")
    axes[0].set_ylabel("总 gRPC 写事件数（log）")
    axes[0].set_title("写事件数 vs 集群规模")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend()

    axes[1].plot(n, [r["baseline_final_convergence_s"] for r in scaling_rows],
                 marker="o", color=C_BASE, label="无 batching")
    axes[1].plot(n, [r["batching_final_convergence_s"] for r in scaling_rows],
                 marker="s", color=C_BATCH, label="有 batching")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("节点数 N（log）")
    axes[1].set_ylabel("最终收敛时间（秒，log）")
    axes[1].set_title("收敛时间 vs 集群规模")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend()
    _finish(fig, os.path.join(outdir, "07_scaling.png"), files)
    return files


def render_all(base_rows, batch_rows, base_waves, batch_waves, base_res, batch_res,
               outdir, sweep_rows=None, scaling_rows=None, prd_rows=None) -> List[str]:
    files: List[str] = []
    files += fig_write_rate(base_rows, batch_rows, outdir)
    files += fig_cumulative(base_rows, batch_rows, outdir, prd_rows=prd_rows)
    files += fig_convergence(base_rows, batch_rows, outdir)
    files += fig_latency(base_waves, batch_waves, outdir)
    files += fig_batch_profile(batch_res, batch_rows, outdir)
    files += fig_sweep(sweep_rows or [], outdir)
    files += fig_scaling(scaling_rows or [], outdir)
    return files
