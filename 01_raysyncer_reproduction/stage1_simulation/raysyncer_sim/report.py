"""Summary JSON + markdown report generation."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Sequence

from .core import SimResult
from .monitor import Recorder


def build_summary(res: SimResult, rec: Recorder, rows: Sequence[Dict[str, Any]],
                  meta: Dict[str, Any]) -> Dict[str, Any]:
    ws = rec.wave_summary()
    peak_total = max((r["link_backlog_total"] for r in rows), default=0)
    peak_link = max((r["link_backlog_max"] for r in rows), default=0)
    peak_gcs_q = max((r["gcs_in_backlog"] for r in rows), default=0)
    peak_gcs_q_at_recv = max((r["gcs_in_backlog_at_recv_max"] for r in rows), default=0)
    peak_lat = max((r["propagation_latency_avg_ms"] for r in rows), default=0.0)
    peak_busy = max((r["gcs_busy_s"] for r in rows), default=0.0)

    return {
        "meta": meta,
        "cluster": {
            "num_nodes": res.num_nodes,
            "ideal_wave1_fanout": res.num_nodes * (res.num_nodes - 1),
        },
        "messages": {
            "inbound_to_gcs": res.total_inbound_messages,
            "accepted_by_gcs": res.total_inbound_messages - res.total_stale_dropped,
            "stale_dropped": res.total_stale_dropped,
            "wave1_updates": res.total_wave1_messages,
            "wave2_updates": res.total_wave2_messages,
            "link_writes_total": res.total_link_writes,
            "push_to_sending_queue_calls": res.total_pushes,
            "pushes_merged_in_place": res.total_inplace_merges,
            "pushes_that_became_a_write": res.total_new_entry_writes,
            "first_time_slot_writes": res.total_slot_writes,
            "wave2_resend_writes": res.total_resend_writes,
            "nodes_contributing_a_resend": int(
                ((res.new_entries_by_src / max(1, res.num_nodes - 1)) - 1.0 > 0).sum()
            ),
            "amplification_writes_per_update": res.amplification,
            "amplification_vs_wave1_only": (
                res.total_link_writes / max(1.0, float(res.num_nodes * (res.num_nodes - 1)))
            ),
        },
        "queues": {
            "peak_total_link_backlog": int(peak_total),
            "peak_single_link_backlog": int(peak_link),
            "peak_gcs_inbound_backlog_seen": int(peak_gcs_q),
            "peak_gcs_inbound_backlog_on_arrival": int(peak_gcs_q_at_recv),
            "gcs_busiest_second_s": round(float(peak_busy), 3),
        },
        "latency": {
            "peak_second_mean_propagation_latency_ms": round(float(peak_lat), 1),
            "wave1": ws["wave1"],
            "wave2": ws["wave2"],
        },
        "convergence": {
            "converged": res.converged,
            "termination_reason": res.termination_reason,
            "final_version_coverage": res.final_version_coverage,
            "uncovered_final_pairs": res.uncovered_final_pairs,
            "wave1_convergence_s": round(res.wave1_convergence_s, 3),
            "final_convergence_s": round(res.final_convergence_s, 3),
            "extra_delay_from_wave2_s": round(
                res.final_convergence_s - res.wave1_convergence_s, 3
            ),
            "slowdown_ratio": round(
                res.final_convergence_s / max(1e-9, res.wave1_convergence_s), 3
            ),
            "wave2_first_emit_s": (
                round(rec.first_wave2_emit_t, 3)
                if rec.first_wave2_emit_t is not None else None
            ),
        },
    }


def _json_safe(value):
    """JSON has no NaN/Infinity; missing times are null."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(data), fh, indent=2, ensure_ascii=False, allow_nan=False)


def _fmt(v: float, nd: int = 2) -> str:
    if v is None or not math.isfinite(v):
        return "未达到/无样本"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{nd}f}"


def _second_table(rows: Sequence[Dict[str, Any]], limit: int = 40) -> str:
    if not rows:
        return "_(no samples)_\n"
    idx = list(range(len(rows)))
    if len(rows) > limit:
        step = max(1, len(rows) // limit)
        idx = list(range(0, len(rows), step))
        if idx[-1] != len(rows) - 1:
            idx.append(len(rows) - 1)
    head = ("| t(s) | updates emitted | GCS received | GCS broadcasts | "
            "GCS->Raylet writes | avg node sent | avg node recv | "
            "GCS inbound backlog | total link backlog | mean prop. latency (ms) |\n")
    head += "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    body = []
    for i in idx:
        r = rows[i]
        body.append(
            f"| {r['second']:.0f} | {r['updates_emitted']:.0f} | {r['gcs_received']:.0f} | "
            f"{r['gcs_broadcasts']:.0f} | {r['link_writes']:.0f} | "
            f"{r['avg_node_sent_per_sec']:.3f} | {r['avg_node_recv_per_sec']:.3f} | "
            f"{r['gcs_in_backlog']:.0f} | {r['link_backlog_total']:.0f} | "
            f"{r['propagation_latency_avg_ms']:.1f} |"
        )
    return head + "\n".join(body) + "\n"


def build_report(summary: Dict[str, Any], rows, figures: List[str],
                 scaling: List[Dict[str, Any]] | None = None,
                 scaling_summary: Dict[str, Any] | None = None) -> str:
    meta = summary["meta"]
    n = summary["cluster"]["num_nodes"]
    m = summary["messages"]
    q = summary["queues"]
    lat = summary["latency"]
    conv = summary["convergence"]
    w1 = lat["wave1"]
    w2 = lat["wave2"]

    out: List[str] = []
    a = out.append

    a(f"# Ray Syncer 传播问题复现报告（N = {n}，无 batching）\n")
    a("对应 ray-project/ray PR **#57641** 的 *Problem / Analysis* 章节：")
    a("在 GCS 与 Raylet 之间复现“**立即广播（push-based, immediate broadcast）**”")
    a("的 Ray Syncer 同步路径，不含该 PR 后续提出的 batching。\n")

    a("## 1. 结论速览\n")
    a(f"- **O(N²) 放大**：{n} 个节点各产生 1 次资源更新，GCS 需要写出 "
      f"**{m['link_writes_total']:,}** 条消息（理想第一波 {n}×{n-1} = "
      f"{n*(n-1):,} 条），"
      f"即每条更新扇出到 N-1 个连接；实测放大倍数 "
      f"**{m['amplification_vs_wave1_only']:.2f}×**（相对纯第一波）。")
    a(f"- **第一波造成队列堆积**：单连接的 `sending_buffer_` 最深堆到 "
      f"**{q['peak_single_link_backlog']:,}** 条（上限 N-1 = {n-1}），"
      f"全集群排队总量峰值 **{q['peak_total_link_backlog']:,}** 条。")
    if meta["wave2_enabled"] and conv["converged"]:
        a(f"- **第二波进入 backlog**：第二波消息入队时，目标连接的积压平均 "
          f"**{_fmt(w2['mean_backlog_at_enqueue'], 1)}** 条（第一波仅 "
          f"**{_fmt(w1['mean_backlog_at_enqueue'], 1)}** 条），P95 达到 "
          f"**{_fmt(w2['p95_backlog_at_enqueue'], 0)}** 条 —— 新更新排在旧积压之后。")
        a(f"- **第二波进一步延迟同步**：第一波的平均传播延迟 "
          f"**{_fmt(w1['mean_end_to_end_s'], 3)} s**，第二波为 "
          f"**{_fmt(w2['mean_end_to_end_s'], 3)} s**（放大 "
          f"**{w2['mean_end_to_end_s']/max(1e-9,w1['mean_end_to_end_s']):.1f}×**）。")
        a(f"- **收敛时间被拉长**：第一波收敛 **{_fmt(conv['wave1_convergence_s'], 1)} s**，"
          f"包含第二波反馈后的最终收敛 **{_fmt(conv['final_convergence_s'], 1)} s**，"
          f"额外 **+{_fmt(conv['extra_delay_from_wave2_s'], 1)} s**"
          f"（{_fmt(conv['slowdown_ratio'], 2)}×）。\n")
    
    elif not meta["wave2_enabled"]:
        a("- **第二波已关闭**：本次只有第一波，不作第二波积压或延迟的结论。")
        a(f"- 最终收敛：**{_fmt(conv['final_convergence_s'])} s**。\n")
    else:
        a("- **本次未达到最终收敛**：不能将最后一次接收时刻当作收敛时间。\n")

    a("## 2. 复现的机制（与 C++ 代码一一对应）\n")
    a("| 真实 Ray 代码 | 仿真实现 | 作用 |")
    a("|---|---|---|")
    a("| `RaySyncer::BroadcastMessage` | `Simulator._on_gcs_in` + `_fanout` | GCS 每收到一条被接受的更新，就遍历**所有** reactor 调用 `PushToSendingQueue` → O(N²) |")
    a("| `NodeState::ConsumeSyncMessage` | `Simulator.gcs_versions` | GCS 侧按 `(node, type)` 版本去重 |")
    a("| `PushToSendingQueue` | `Reactor.push` | 按 `(node_id, message_type)` 存入 `sending_buffer_`，旧的同 key 更新被原地替换 |")
    a("| `StartSend` / `sending_` | `Reactor.start_write` | **每条连接同时只有 1 个写在空中**，其余留在 buffer → 队头阻塞 |")
    a("| gRPC write 完成回调 | GCS FIFO 执行器（`gcs_write_us`） | 单线程 GCS 上，收包扇出与写完成回调竞争同一 CPU |")
    a("| Raylet 上报 (100 ms 周期 / 事件驱动) | 第一波 `start_wave1` + 第二波 `_check_milestones` | 波次来源 |\n")

    a("## 3. 监控指标\n")
    a("### 3.1 每秒 update / GCS 收 / GCS 广播 / 平均节点收发\n")
    a("“节点发送”= Raylet 上行到 GCS 的资源更新数；“节点接收”= 该秒内"
      "Raylet 从 GCS 收到并写入本地资源视图的消息数（= 完成的 link write）。\n")
    a(_second_table(rows))

    a("\n### 3.2 队列长度\n")
    a(f"- 单连接 `sending_buffer_` 峰值：**{q['peak_single_link_backlog']:,}** 条"
      f"（理论上限 N-1 = {n-1}）")
    a(f"- 全集群排队总量峰值：**{q['peak_total_link_backlog']:,}** 条")
    a(f"- GCS 入站积压（已收未扇出）峰值：**{q['peak_gcs_inbound_backlog_seen']:,}** 条")
    a(f"- 新消息入队时看到的积压：第一波均值 {_fmt(w1['mean_backlog_at_enqueue'], 1)} / "
      f"P95 {_fmt(w1['p95_backlog_at_enqueue'], 0)}；第二波均值 "
      f"{_fmt(w2['mean_backlog_at_enqueue'], 1)} / P95 {_fmt(w2['p95_backlog_at_enqueue'], 0)}\n")

    a("### 3.3 传播延迟\n")
    a("延迟 = 消息产生 → 目标 Raylet 本地资源视图被更新（端到端）。\n")
    a("| 指标 | 第一波 | 第二波 |")
    a("|---|---:|---:|")
    a(f"| 均值传播延迟 (s) | {_fmt(w1['mean_end_to_end_s'], 3)} | {_fmt(w2['mean_end_to_end_s'], 3)} |")
    a(f"| P50 传播延迟 (s) | {_fmt(w1['p50_end_to_end_s'], 3)} | {_fmt(w2['p50_end_to_end_s'], 3)} |")
    a(f"| P95 传播延迟 (s) | {_fmt(w1['p95_end_to_end_s'], 3)} | {_fmt(w2['p95_end_to_end_s'], 3)} |")
    a(f"| 最大传播延迟 (s) | {_fmt(w1['max_end_to_end_s'], 3)} | {_fmt(w2['max_end_to_end_s'], 3)} |")
    a(f"| 均值排队等待 (s) | {_fmt(w1['mean_link_queue_wait_s'], 3)} | {_fmt(w2['mean_link_queue_wait_s'], 3)} |")
    a(f"| P95 排队等待 (s) | {_fmt(w1['p95_link_queue_wait_s'], 3)} | {_fmt(w2['p95_link_queue_wait_s'], 3)} |")
    a(f"| 均值 GCS 扇出排队 (s) | {_fmt(w1['mean_gcs_queue_delay_s'], 3)} | {_fmt(w2['mean_gcs_queue_delay_s'], 3)} |\n")

    a("### 3.4 收敛时间\n")
    a(f"- 最终版本覆盖率：**{conv['final_version_coverage']:.2%}**；"
      f"未覆盖源—目标对：**{conv['uncovered_final_pairs']}**；"
      f"结束原因：`{conv['termination_reason']}`。")
    a(f"- 第一波收敛（所有 Raylet 都拿到第一波全部更新）：**{_fmt(conv['wave1_convergence_s'])} s**")
    a(f"- 最终收敛（所有 Raylet 都拿到最终版本）：**{_fmt(conv['final_convergence_s'])} s**")
    a(f"- 第二波造成的额外延迟：**+{_fmt(conv['extra_delay_from_wave2_s'])} s**\n")

    a("### 3.5 消息量拆解：多出来的这一倍是从哪来的？\n")
    a(f"GCS 每接受 1 条更新，就要对 N-1 个连接各调用一次 `PushToSendingQueue`；"
      f"本次运行共 {m['accepted_by_gcs']:,} 条被接受的更新（第一波 {m['wave1_updates']:,} + "
      f"第二波 {m['wave2_updates']:,}），因此扇出推送 **{m['push_to_sending_queue_calls']:,}** 次。"
      "推送的结果只有两种：\n")
    a("| 推送结果 | 次数 | 代价 |")
    a("|---|---:|---|")
    a(f"| 该 `(node_id, message_type)` 槽位**仍在 `sending_buffer_` 里** → 原地合并（新版本顶掉旧版本） | "
      f"{m['pushes_merged_in_place']:,} "
      f"({m['pushes_merged_in_place']/max(1,m['push_to_sending_queue_calls'])*100:.1f}%) | **0 次写** |")
    a(f"| 该槽位**已经被写出去** → 新建一个条目 | {m['pushes_that_became_a_write']:,} | "
      f"**1 次写**，并且排在整条 backlog 的队尾 |")
    a("\n所以“实际写出多少条”完全由第二类决定：\n")
    a("| 写出的消息 | 条数 | 说明 |")
    a("|---|---:|---|")
    a(f"| 为每个 (源节点, 目标连接) 建立第一个槽位 | "
      f"{m['first_time_slot_writes']:,} | = N(N-1)，这就是“理想第一波扇出” |")
    a(f"| 槽位已写出后，第二波的新版本重新入队 | **{m['wave2_resend_writes']:,}** | "
      f"**多出来的部分** |")
    a(f"| 合计 | **{m['link_writes_total']:,}** | "
      f"{m['link_writes_total']/max(1,m['first_time_slot_writes']):.2f}× 理想值 |")
    if meta["wave2_enabled"]:
        a(f"\n这 {m['wave2_resend_writes']:,} 条重发的来源：每个节点在恢复期会发出 "
          f"{len(meta['wave2_milestones'])} 个新版本，绝大多数版本到达时它的槽位还排在队列里"
          f"（免费合并）；但每个节点**总有一次**是在自己的槽位已经被写出去之后才发出的，"
          f"这个版本只能重新排到队尾。实测 **{m['nodes_contributing_a_resend']}/{meta['num_nodes']}** "
          f"个节点各贡献 1 条这样的重发，其余节点贡献 0 条"
          f"（平均 {m['wave2_resend_writes']/max(1,m['first_time_slot_writes']):.2f} 条/节点/连接）。\n")
    
    a("## 4. 图表\n")
    for f in sorted(figures, key=os.path.basename):
        a(f"![{os.path.basename(f)}]({os.path.basename(f)})\n")

    if scaling:
        a("\n## 5. 规模扩展：O(N²) 验证\n")
        a("| N | 第一波理想扇出 N(N-1) | 实际写出消息数 | 实际/理想 | 第一波收敛 (s) | 最终收敛 (s) |")
        a("|---:|---:|---:|---:|---:|---:|")
        for s in scaling:
            ideal = s["num_nodes"] * (s["num_nodes"] - 1)
            a(f"| {s['num_nodes']} | {ideal:,} | {s['total_link_writes']:,} | "
              f"{s['total_link_writes']/ideal:.2f}× | "
              f"{s['wave1_convergence_s']:.2f} | {s['final_convergence_s']:.2f} |")
        if scaling_summary:
            a(f"\n- 增长指数拟合：**{scaling_summary.get('exponent', float('nan')):.2f}**"
              f"（≈2 即 O(N²)）")
            a(f"- 二次拟合系数：{scaling_summary.get('quad_coef', 0.0):.3e} N²\n")

    a("\n## 6. 参数与标定\n")
    a("| 参数 | 取值 | 说明 |")
    a("|---|---:|---|")
    a(f"| `num_nodes` | {meta['num_nodes']} | 集群规模 |")
    a(f"| `gcs_fixed_us` | {meta['gcs_fixed_us']} | 每条入站消息的固定开销 |")
    a(f"| `gcs_fanout_us` | {meta['gcs_fanout_us']} | 每个 reactor 的扇出开销（O(N) 来源）|")
    a(f"| `gcs_write_us` | {meta['gcs_write_us']} | 每次 link 写在 GCS 事件循环上的开销（瓶颈）|")
    a(f"| `prop_us` | {meta['prop_us']} | GCS→Raylet 单跳时延 |")
    a(f"| `uplink_us` | {meta['uplink_us']} | Raylet→GCS 单跳时延 |")
    a(f"| `wave2_milestones` | {meta['wave2_milestones']} | 第二波触发点（本地视图完成度）|")
    a("\n标定：令 `gcs_write_us` 使得 N=1000 时第一波扇出耗时 ≈ 150 s，"
      "与 PR 中 1K 节点第一波（0–10 s 突发、约 150 s 完成同步）量级一致。\n")

    a("## 7. 局限\n")
    a("- 这是**传播层**的复现：只建模 Ray Syncer 的扇出/队列/写路径，不跑真实 Ray，"
      "不包含任务调度、PG 资源预约、gRPC/TCP 细节。")
    a("- `gcs_write_us` 是**等效服务时间**，把单线程 GCS 上的 gRPC 写、回调、"
      "内存与竞争效应合成为一个参数。")
    a("- 单连接 buffer 在真实代码里是 `flat_hash_map`（顺序不确定）；"
      "这里用插入序 FIFO 以复现 PR 描述的“新更新排在积压之后”。\n")
    return "\n".join(out)


def write_report(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)

