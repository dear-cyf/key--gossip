"""Stage-2 summary + markdown report: baseline vs batching."""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Sequence


def _run_block(res, waves) -> Dict[str, Any]:
    return {
        "converged": res.converged,
        "termination_reason": res.termination_reason,
        "final_version_coverage": res.final_version_coverage,
        "uncovered_final_pairs": res.uncovered_final_pairs,
        "write_events": int(res.total_link_writes),
        "messages_written": int(res.total_messages_written),
        "mean_batch_size": float(res.mean_batch_size),
        "inbound_messages": int(res.total_inbound_messages),
        "stale_dropped": int(res.total_stale_dropped),
        "pushes": int(res.total_pushes),
        "inplace_merges": int(res.total_inplace_merges),
        "new_entry_writes": int(res.total_new_entry_writes),
        "wave1_convergence_s": float(res.wave1_convergence_s),
        "final_convergence_s": float(res.final_convergence_s),
        "batch_count": int(res.batch_sizes.size),
        "batch_size_max": int(res.batch_sizes.max()) if res.batch_sizes.size else 0,
        "trigger_counts": dict(res.trigger_counts),
        "wave1": waves.get("wave1", {}),
        "wave2": waves.get("wave2", {}),
    }


def build_summary(base_res, base_waves, batch_res, batch_waves, meta,
                  prd_res=None, prd_waves=None) -> Dict[str, Any]:
    base = _run_block(base_res, base_waves)
    bat = _run_block(batch_res, batch_waves)
    summary = {
        "meta": dict(meta),
        "baseline": base,
        "batching": bat,
        "comparison": {
            "write_event_reduction_x": (
                base["write_events"] / bat["write_events"] if bat["write_events"] else 0.0
            ),
            "messages_written_reduction_x": (
                base["messages_written"] / bat["messages_written"]
                if bat["messages_written"] else 0.0
            ),
            "wave1_convergence_speedup_x": (
                base["wave1_convergence_s"] / bat["wave1_convergence_s"]
                if bat["wave1_convergence_s"] else 0.0
            ),
            "final_convergence_speedup_x": (
                base["final_convergence_s"] / bat["final_convergence_s"]
                if bat["final_convergence_s"] else 0.0
            ),
            "wave1_latency_speedup_x": (
                base["wave1"]["mean_end_to_end_s"] / bat["wave1"]["mean_end_to_end_s"]
                if bat["wave1"].get("mean_end_to_end_s") else 0.0
            ),
            "wave2_latency_speedup_x": (
                base["wave2"]["mean_end_to_end_s"] / bat["wave2"]["mean_end_to_end_s"]
                if bat["wave2"].get("mean_end_to_end_s") else 0.0
            ),
        },
    }
    if prd_res is not None:
        summary["pr_default"] = _run_block(prd_res, prd_waves or {})
        summary["comparison"]["pr_default_vs_baseline_write_reduction_x"] = (
            base["write_events"] / prd_res.total_link_writes
            if prd_res.total_link_writes else 0.0
        )
    return summary


def _json_safe(value):
    """JSON has no NaN/Infinity; missing times are null."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(obj), fh, indent=2, ensure_ascii=False, default=float, allow_nan=False)


def write_report(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _fmt(x: float, nd: int = 3) -> str:
    return f"{x:,.{nd}f}"


def _sweep_table(sweep: Optional[Sequence[Dict[str, Any]]]) -> str:
    if not sweep:
        return "_（未运行参数扫描）_\n"
    out = [
        "| batch size | 超时 (ms) | 写事件数 | 批次数 | 平均批大小 | 最终收敛 (s) | 相对基线加速 |",
        "|---|---|---|---|---|---|---|",
    ]
    base = float(sweep[0]["baseline_final_convergence_s"]) or 1.0
    for r in sweep:
        out.append(
            f"| {r['batch_size']} | {r['batch_delay_ms']:.0f} | "
            f"{r['write_events']:,} | {r['batch_count']:,} | "
            f"{r['mean_batch_size']:.1f} | {r['final_convergence_s']:.2f} | "
            f"{base / max(1e-9, r['final_convergence_s']):.2f}× |"
        )
    return "\n".join(out) + "\n"


def _scaling_table(scaling: Optional[Sequence[Dict[str, Any]]]) -> str:
    if not scaling:
        return "_（未运行规模扩展）_\n"
    arm = scaling[0].get("arm", "batching")
    if arm == "pr_default":
        label = "post-PR 默认 (1 / 0 ms)"
    else:
        label = (f"batching ({scaling[0].get('batch_size', 1000)} / "
                 f"{scaling[0].get('batch_delay_ms', 500):g} ms)")
    out = [
        f"| N | 基线写事件 | {label} 写事件 | 缩减倍数 | 基线收敛 (s) | {label} 收敛 (s) | 加速 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in scaling:
        red = r["baseline_write_events"] / max(1, r["batching_write_events"])
        sp = r["baseline_final_convergence_s"] / max(1e-9, r["batching_final_convergence_s"])
        out.append(
            f"| {r['num_nodes']} | {r['baseline_write_events']:,} | "
            f"{r['batching_write_events']:,} | {red:.1f}× | "
            f"{r['baseline_final_convergence_s']:.2f} | "
            f"{r['batching_final_convergence_s']:.2f} | {sp:.1f}× |"
        )
    return "\n".join(out) + "\n"


def build_report(summary: Dict[str, Any], sweep, scaling, figures: List[str]) -> str:
    m = summary["meta"]
    base = summary["baseline"]
    bat = summary["batching"]
    cmp_ = summary["comparison"]

    trig = bat.get("trigger_counts", {})
    trig_txt = "、".join(f"{k} {v:,}" for k, v in sorted(trig.items())) or "—"

    lines: List[str] = []
    lines.append("# 阶段二：Ray Syncer Sync Message Batching 仿真对照\n")
    lines.append(
        "本报告用与阶段一**完全相同的离散事件模型**（同一 GCS 单线程执行器、同一 "
        "O(N²) 扇出、同一第二波正反馈负载）对 ray-project/ray PR #57641 "
        "提出的 batching 做对照。**阶段一的代码与结果未被修改**，本目录是独立副本。\n"
    )
    lines.append(
        f"- 集群规模：N = **{m['num_nodes']}**，seed = {m['seed']}\n"
        f"- 基线（无 batching）：`max_batch_size = 1`, `max_batch_delay_ms = 0`\n"
        f"- batching：`max_batch_size = {m['batch_size']}`, "
        f"`max_batch_delay_ms = {m['batch_delay_ms']:g}`\n"
        f"- 每条链路写开销 `gcs_write_us = {m['gcs_write_us']:g}`，"
        f"每消息扇出开销 `gcs_fanout_us = {m['gcs_fanout_us']:g}`\n"
    )

    lines.append("\n## 1. 结论速览\n")
    lines.append("| 指标 | 无 batching | 有 batching | 变化 |\n")
    lines.append("|---|---|---|---|\n")
    lines.append(
        f"| GCS→Raylet gRPC 写事件 | {base['write_events']:,} | {bat['write_events']:,} | "
        f"**减少 {cmp_['write_event_reduction_x']:.1f}×** |\n"
    )
    lines.append(
        f"| 实际发出的 RaySyncMessage 数 | {base['messages_written']:,} | "
        f"{bat['messages_written']:,} | 基本持平（批内合并后略增） |\n"
    )
    lines.append(
        f"| 第一波收敛 | {base['wave1_convergence_s']:.2f} s | "
        f"{bat['wave1_convergence_s']:.2f} s | "
        f"**{cmp_['wave1_convergence_speedup_x']:.1f}× 更快** |\n"
    )
    lines.append(
        f"| 最终收敛 | {base['final_convergence_s']:.2f} s | "
        f"{bat['final_convergence_s']:.2f} s | "
        f"**{cmp_['final_convergence_speedup_x']:.1f}× 更快** |\n"
    )
    lines.append(
        f"| 第一波平均传播延迟 | {base['wave1']['mean_end_to_end_s']*1000:.1f} ms | "
        f"{bat['wave1']['mean_end_to_end_s']*1000:.1f} ms | "
        f"含 {m['batch_delay_ms']:g} ms 等待，见第 5 节 |\n"
    )
    lines.append(
        f"| 第二波平均传播延迟 | {base['wave2']['mean_end_to_end_s']*1000:.1f} ms | "
        f"{bat['wave2']['mean_end_to_end_s']*1000:.1f} ms | "
        f"{cmp_['wave2_latency_speedup_x']:.1f}× 更低 |\n"
    )
    lines.append(
        f"| 批次数 / 平均批大小 | 每条消息一批（{base['messages_written']:,}） | "
        f"{bat['batch_count']:,} / {bat['mean_batch_size']:.1f} | — |\n"
    )

    prd = summary.get("pr_default")
    if prd:
        lines.append(
            "\n**注意：合并后的默认配置本身就是“改进后”**。PR 把 "
            "`gcs_resource_broadcast_max_batch_size` 默认设为 1、"
            "`max_batch_delay_ms` 默认设为 0，看起来像“关闭 batching”，"
            "但 `StartSend()` 已经改成把整个 `sending_buffer_` 打包——它只是不等超时。\n\n"
            f"在同样的 {m['num_nodes']} 节点负载下，这个默认配置已经把写事件从 "
            f"{base['write_events']:,} 降到 **{prd['write_events']:,}**"
            f"（{summary['comparison'].get('pr_default_vs_baseline_write_reduction_x', 0):.1f}×），"
            f"最终收敛 {prd['final_convergence_s']:.2f} s"
            f"（基线 {base['final_convergence_s']:.2f} s）；"
            f"再叠加 {m['batch_delay_ms']:g} ms 超时后进一步降到 "
            f"{bat['write_events']:,} 次写、{bat['final_convergence_s']:.2f} s。\n"
        )

    lines.append("\n## 2. 机制对照\n")
    lines.append(
        "阶段一（无 batching，PR base `aaeb5f8f`）：\n\n"
        "```cpp\n"
        "// PushToSendingQueue()\n"
        "sending_buffer_[{node_id, message_type}] = message;\n"
        "StartSend();               // 立刻尝试写\n\n"
        "// StartSend()\n"
        "if (sending_) return;      // 每条链路同时只有 1 个写在飞\n"
        "auto it = sending_buffer_.begin();\n"
        "Send(std::move(it->second), ...);   // 一次只写 1 条消息\n"
        "```\n"
    )
    lines.append(
        "阶段二（batching，PR head `d42cf042`）：\n\n"
        "```cpp\n"
        "// PushToSendingQueue()\n"
        "sending_buffer_[{node_id, message_type}] = message;\n"
        "if (sending_buffer_.size() >= max_batch_size_ || max_batch_delay_ms_.count() == 0) {\n"
        "  batch_timer_.cancel();\n"
        "  StartSend();                       // 达到 batch size：立即发\n"
        "} else if (!batch_timer_active_) {\n"
        "  batch_timer_.expires_after(max_batch_delay_ms_);   // 否则等到超时\n"
        "  batch_timer_.async_wait(... StartSend ...);\n"
        "}\n\n"
        "// StartSend()：把整个 sending_buffer_ 打包成一个 RaySyncMessageBatch\n"
        "for (const auto &[key, message] : sending_buffer_) *batch->add_messages() = *message;\n"
        "sending_buffer_.clear();\n"
        "StartWrite(batch);                       // 1 次 gRPC 写，携带 K 条消息\n"
        "```\n"
    )
    lines.append(
        "仿真里一一对应：`Reactor.push()` 是 `PushToSendingQueue`，`Reactor.start_write()` "
        "是 `StartSend`，`E_BATCH_TIMER` 事件是 `batch_timer_.async_wait`。"
        "除了发送路径，其余（扇出、版本去重、投递、收敛统计）与阶段一逐行一致。\n"
    )

    lines.append("\n## 3. 写路径与排队\n")
    lines.append(
        f"- 基线：GCS 收到 {base['inbound_messages']:,} 条入站更新，向 N-1 个 reactor 推送 "
        f"{base['pushes']:,} 次；其中 {base['inplace_merges']:,} 次在 `sending_buffer_` 里原地合并，"
        f"最终实际产生 **{base['write_events']:,} 次写**。\n"
        f"- batching：同样的 {bat['pushes']:,} 次推送、{bat['inplace_merges']:,} 次原地合并，"
        f"但被合并进 **{bat['batch_count']:,} 个批次**，只产生 **{bat['write_events']:,} 次写**，"
        f"一次写平均携带 {bat['mean_batch_size']:.1f} 条消息（最大 {bat['batch_size_max']:,}）。\n"
        f"- 批次触发原因：{trig_txt}。\n"
    )

    lines.append("\n## 4. 传播延迟与收敛\n")
    lines.append("| 波次 | 基线平均延迟 | batching 平均延迟 | 基线 P95 | batching P95 |\n")
    lines.append("|---|---|---|---|---|\n")
    for key, label in (("wave1", "第一波"), ("wave2", "第二波")):
        lines.append(
            f"| {label} | {base[key]['mean_end_to_end_s']*1000:.1f} ms | "
            f"{bat[key]['mean_end_to_end_s']*1000:.1f} ms | "
            f"{base[key]['p95_end_to_end_s']*1000:.1f} ms | "
            f"{bat[key]['p95_end_to_end_s']*1000:.1f} ms |\n"
        )
    lines.append(
        f"\n- 每条连接 `sending_buffer_` 峰值排队：基线 **{base['wave1'].get('max_backlog_at_enqueue', 0):.0f}**"
        f"（入队时观测），batching **{bat['wave1'].get('max_backlog_at_enqueue', 0):.0f}**。\n"
    )

    lines.append("\n## 5. 参数扫描（batch size × timeout）\n")
    lines.append(_sweep_table(sweep))
    lines.append(
        "\n> 超时越大，合并越彻底、写事件越少，但每条更新的等待也越久；"
        "batch size 越小越早触发写，效果介于两者之间。\n"
    )

    lines.append("\n## 6. 规模扩展（N 变化）\n")
    lines.append(_scaling_table(scaling))
    lines.append(
        "\n> 无 batching 时写事件数 ≈ N(N-1)，随 N 二次增长；batching 后每个连接把整批合并成一次写，"
        "写事件数退化为 ~O(N)（每轮每个连接 1 次写），这是 PR 在 1K 节点上把 10 分钟压到 1 分钟量级的直接原因。\n"
    )

    if figures:
        lines.append("\n## 7. 图\n")
        for f in figures:
            lines.append(f"- `{f.split(chr(92))[-1].split('/')[-1]}`\n")

    lines.append("\n## 8. 局限\n")
    lines.append(
        "1. 仍然只建模传播层，不含真实任务调度、PG 资源预约、gRPC/TCP 与序列化细节。\n"
        "2. 批次按“一次 gRPC 写”计费，额外的单条序列化开销由 `batch_per_msg_us` 控制，默认 0；"
        "调大它可以做敏感性分析。\n"
        "3. 需要注意：**合并后的默认配置 `max_batch_size=1 / delay=0` 并不等价于 pre-PR 行为**。"
        "`StartSend()` 已经会打包整个 `sending_buffer_`，只是不等待超时，因此它本身就能把写事件减少 "
        f"{summary['comparison'].get('pr_default_vs_baseline_write_reduction_x', 0):.0f}×；"
        "本报告第 1 节的“无 batching”一列用的是 `post_pr_send_path=False`（真正的 pre-PR 语义）。\n"
        "4. 真实 `sending_buffer_` 是 hash map、批内顺序不确定；仿真用插入序，偏保守。\n"
    )
    return "".join(lines)
