"""Star propagation with PR #57641's per-stream sending-buffer semantics.

One write may be active per stream. While it is active, new versions replace
older entries in the pending buffer, even after a size/timeout trigger. StartSend
packs the entire buffer; OnWriteDone immediately attempts the next send. The
existing abstract write-service completion remains the callback boundary, not
an acknowledgement of remote receipt or a detailed gRPC transport model.
"""

from __future__ import annotations

import heapq
import itertools
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import US, SimConfig

# Share transport/workload semantics with stage 1, including standalone CLI use.
import sys
from pathlib import Path
_stage1 = str(Path(__file__).resolve().parents[3] / "01_raysyncer_reproduction/stage1_simulation")
if _stage1 not in sys.path:
    sys.path.insert(0, _stage1)
from raysyncer_sim.costs import Network, ResourceWorkload, CALLBACK

MSG_RESOURCE_VIEW = 0  # ray::rpc::syncer::MessageType::RESOURCE_VIEW

E_GCS_IN = "gcs_in"
E_FANOUT = "fanout"
E_WRITE = "write"
E_EXEC_DONE = "exec_done"
E_RECV = "recv"
E_BATCH_TIMER = "batch_timer"
E_SAMPLE = "sample"


class Message:
    """A single ``ResourceViewSyncMessage`` travelling through the cluster."""

    __slots__ = (
        "resources",
        "mid",
        "src",
        "version",
        "wave",
        "rnd",
        "emit_t",
        "gcs_recv_t",
        "gcs_in_q_at",
        "fanout_start_t",
    )

    def __init__(self, mid: int, src: int, version: int, wave: int, rnd: int,
                 emit_t: float):
        self.resources = None
        self.mid = mid
        self.src = src
        self.version = version
        self.wave = wave
        self.rnd = rnd
        self.emit_t = emit_t
        self.gcs_recv_t: Optional[float] = None
        self.gcs_in_q_at: int = -1
        self.fanout_start_t: Optional[float] = None


class Reactor:
    """One ``RaySyncerBidiReactor``: the GCS -> Raylet direction of a stream."""

    __slots__ = ("node", "buffer", "inflight", "last_version", "writes",
                 "last_delivery_t", "msgs_written", "batch_timer_active",
                 "timer_seq", "pending_trigger", "batch_sizes",
                 "trigger_counts")

    def __init__(self, node: int):
        self.node = node
        # sending_buffer_: ordered by insertion so the drain order is a
        # deterministic FIFO.  Re-assigning an existing key keeps its slot,
        # exactly like re-inserting into the C++ hash map.
        self.buffer: "OrderedDict[Any, Tuple[Message, float, int]]" = OrderedDict()
        # sending_: at most one write in flight.
        self.inflight: Optional[List[Tuple[Message, float, int]]] = None
        self.last_version: Dict[Tuple[int, int], int] = {}
        # Number of gRPC write *events* (a batch counts once).
        self.writes = 0
        # Number of RaySyncMessage objects carried by those write events.
        self.msgs_written = 0
        self.last_delivery_t = -1.0
        # Batching state (PR #57641): arm-once timer + trigger accounting.
        self.batch_timer_active = False
        self.timer_seq = 0
        self.pending_trigger: Optional[str] = None
        self.batch_sizes: List[int] = []
        self.trigger_counts: Dict[str, int] = {}

    # -- PushToSendingQueue -------------------------------------------------
    def push(self, msg: Message, enq_t: float, sim: "Simulator") -> bool:
        if msg.src == self.node:
            # Never send a node its own update back (matches the C++ filter).
            return False
        key = (msg.src, MSG_RESOURCE_VIEW)
        if msg.version <= self.last_version.get(key, -1):
            return False  # stale version, dropped
        self.last_version[key] = msg.version

        backlog = len(self.buffer)
        if sim.cfg.gcs_reactor_dedup:
            if key in self.buffer:
                # Same (node_id, message_type) is still queued: the newer
                # version merges into the existing slot, so it costs no
                # additional write -- only the content gets fresher.
                sim.inplace_merges += 1
            else:
                # The slot was already flushed to this connection, so the new
                # version has to be appended behind the whole backlog.
                sim.new_entry_writes += 1
                sim.new_entries_by_src[msg.src] += 1
            self.buffer[key] = (msg, enq_t, backlog)
        else:
            sim.new_entry_writes += 1
            sim.new_entries_by_src[msg.src] += 1
            self.buffer[(msg.src, MSG_RESOURCE_VIEW, msg.mid)] = (msg, enq_t, backlog)
        sim.total_pushes += 1

        # ---- PR #57641 batching hook (stage 1 called StartSend() here) ---- #
        cfg = sim.cfg
        eager = (
            not cfg.post_pr_send_path
            or not cfg.batching_active
            or len(self.buffer) >= cfg.max_batch_size
        )
        if eager:
            # Buffer reached the size limit (or batching is disabled): cancel
            # any armed timer and write now.
            self.cancel_timer()
            self.pending_trigger = (
                "size"
                if cfg.batching_active and len(self.buffer) >= cfg.max_batch_size
                else "immediate"
            )
            self.start_write(sim)
        elif not self.batch_timer_active:
            # Arm the batch timer.  Later pushes do NOT restart it, matching
            # ``batch_timer_.expires_after()`` being called only when the
            # timer is inactive.
            self.batch_timer_active = True
            self.timer_seq += 1
            sim._push(sim.now + cfg.max_batch_delay_ms * 1e-3,
                      E_BATCH_TIMER, (self.node, self.timer_seq))
        return True

    # -- StartSend ----------------------------------------------------------
    def start_write(self, sim: "Simulator") -> None:
        """PR StartSend: leave the mutable buffer intact while sending_."""
        if self.inflight is not None:
            return
        if not self.buffer:
            self.pending_trigger = None
            return
        # In the PR, max_batch_size is an eager-send threshold, not a hard cap.
        # Entries can accumulate beyond it while sending_ is true.
        limit = len(self.buffer) if sim.cfg.post_pr_send_path else 1
        batch = [self.buffer.popitem(last=False)[1]
                 for _ in range(limit)]
        trigger = self.pending_trigger or "immediate"
        self.pending_trigger = None
        self.inflight = batch
        self.writes += 1
        self.msgs_written += len(batch)
        self.batch_sizes.append(len(batch))
        self.trigger_counts[trigger] = self.trigger_counts.get(trigger, 0) + 1
        if hasattr(sim.rec, "on_seal"):
            sim.rec.on_seal(sim.now, self.node, batch, trigger)
        sim.submit_write(self, batch, sim.now, trigger)

    def cancel_timer(self) -> None:
        """Mirror ``batch_timer_.cancel(); batch_timer_active_ = false;``."""
        if self.batch_timer_active:
            self.batch_timer_active = False
            # Invalidate the pending timer event.
            self.timer_seq += 1


@dataclass
class SimResult:
    """Everything the analysis stage needs, in compact form."""

    cfg: SimConfig
    num_nodes: int
    wall_sim_time: float
    # gRPC write events on the GCS -> Raylet links (a batch counts once).
    total_link_writes: int
    # RaySyncMessage objects carried by those writes.
    total_messages_written: int
    total_inbound_messages: int
    total_stale_dropped: int
    total_pushes: int
    total_new_entry_writes: int
    total_slot_writes: int
    total_resend_writes: int
    total_inplace_merges: int
    new_entries_by_src: np.ndarray
    total_wave1_messages: int
    total_wave2_messages: int
    wave1_convergence_s: float
    final_convergence_s: float
    wave1_last_delivery_s: float
    per_dst_writes: np.ndarray
    per_dst_messages: np.ndarray
    last_delivery_per_dst: np.ndarray
    first_recv: np.ndarray
    last_recv: np.ndarray
    received_versions: np.ndarray
    final_versions: np.ndarray
    converged: bool
    final_version_coverage: float
    uncovered_final_pairs: int
    termination_reason: str
    batch_sizes: np.ndarray
    trigger_counts: Dict[str, int]

    @property
    def amplification(self) -> float:
        """Messages actually written per inbound message (approaches N-1)."""
        if self.total_inbound_messages == 0:
            return 0.0
        return self.total_link_writes / self.total_inbound_messages

    @property
    def mean_batch_size(self) -> float:
        if self.batch_sizes.size == 0:
            return 0.0
        return float(self.batch_sizes.mean())


class Simulator(ResourceWorkload):
    """Single GCS + N Raylets, discrete-event, microsecond resolution."""

    def __init__(self, cfg: SimConfig, recorder=None):
        self.cfg = cfg
        self.rec = recorder
        self.N = cfg.num_nodes
        self.net = Network(self)
        self.resource_state = [0] * self.N
        self.emitted_wave1 = 0
        self.emitted_wave2 = 0
        self.node_proc_free = [0.0] * self.N
        self.processing_wait_s = 0.0
        self.processing_wait_max_s = 0.0
        self.executor_wait_s = 0.0
        self.executor_wait_max_s = 0.0
        self.rng = random.Random(cfg.seed)

        self.heap: List[Tuple[float, int, str, Any]] = []
        self._seq = itertools.count()
        self.n_real = 0
        self.now = 0.0
        self._mid = itertools.count()

        self.reactors = [Reactor(i) for i in range(self.N)]

        # Single-threaded GCS executor: FIFO of (task, cost).
        self.exec_q: List[Tuple[Tuple, float]] = []
        self.exec_busy = False
        self.exec_start = 0.0
        self.gcs_in_backlog = 0

        self.gcs_versions = [-1] * self.N
        self.local_version = [0] * self.N

        # Convergence bookkeeping.
        self.first_recv = np.full((self.N, self.N), -1.0)
        self.last_recv = np.full((self.N, self.N), -1.0)
        # Destination x origin: versions actually applied, not merely queued.
        self.received_versions = np.full((self.N, self.N), -1, dtype=np.int64)
        self.latest_version_recv = np.full((self.N, self.N), -1.0)
        self.termination_reason = "event_queue_exhausted"
        self.w1_count = [0] * self.N
        self.ms_idx = [0] * self.N

        self.milestones: List[List[float]] = []
        base = list(cfg.wave2_milestones)
        for node in range(self.N):
            jit = 1.0
            if cfg.wave2_jitter > 0:
                jit = 1.0 + cfg.wave2_jitter * (self.rng.random() - 0.5)
            ms = [m * jit for m in base]
            if node == cfg.driver_node and cfg.driver_first_milestone > 0:
                ms = [cfg.driver_first_milestone] + ms
            self.milestones.append(ms)

        self.total_stale_dropped = 0
        self.accepted = 0
        # Write accounting: a connection holds at most N-1 slots (one per
        # source node).  Pushes that find their slot already flushed create an
        # *extra* write; pushes that find the slot still queued merge in place.
        self.total_pushes = 0
        self.inplace_merges = 0
        self.new_entry_writes = 0
        self.new_entries_by_src = np.zeros(self.N, dtype=np.int64)

        # Effective per-connection write cost on the shared GCS executor.
        j = cfg.link_write_jitter
        self.write_cost = [
            cfg.gcs_write_us * US
            * (1.0 + (j * (self.rng.random() - 0.5) if j > 0 else 0.0))
            for _ in range(self.N)
        ]

    # ------------------------------------------------------------------ #
    # Event plumbing
    # ------------------------------------------------------------------ #
    def _push(self, t: float, kind: str, payload: Any) -> None:
        heapq.heappush(self.heap, (t, next(self._seq), kind, payload))
        self.n_real += 1

    def _push_sample(self, t: float) -> None:
        heapq.heappush(self.heap, (t, next(self._seq), E_SAMPLE, None))

    def submit_write(self, reactor: Reactor, batch: List[Tuple[Message, float, int]],
                     start_t: float, trigger: str) -> None:
        # One gRPC write slot per batch, plus an optional per-message payload
        # cost (``batch_per_msg_us``, default 0).
        cost = self.write_cost[reactor.node] + (
            self.cfg.batch_per_msg_us * US * len(batch)
        )
        self.exec_q.append(
            ((E_WRITE, reactor.node, batch, start_t, trigger), cost + (self.cfg.message_bytes * len(batch) + self.cfg.frame_overhead_bytes) * self.cfg.serialize_us_per_byte * US, self.now)
        )
        self._pump_exec()

    def _submit_fanout(self, msg: Message) -> None:
        cost = (self.cfg.gcs_fixed_us + self.cfg.gcs_fanout_us * (self.N - 1)) * US
        self.gcs_in_backlog += 1
        self.exec_q.append(((E_FANOUT, msg), cost, self.now))
        self._pump_exec()

    def _pump_exec(self) -> None:
        if self.exec_busy or not self.exec_q:
            return
        task, cost, queued = self.exec_q.pop(0)
        wait = self.now - queued
        self.executor_wait_s += wait
        self.executor_wait_max_s = max(self.executor_wait_max_s, wait)
        self.exec_busy = True
        self.exec_start = self.now
        if task[0] == E_WRITE and hasattr(self.rec, "on_write_service"):
            self.rec.on_write_service(task[1], task[3], self.now, self.now + cost)
        self._push(self.now + cost, E_EXEC_DONE, task)

    # ------------------------------------------------------------------ #
    # Workload
    # ------------------------------------------------------------------ #
    def emit(self, node: int, wave: int, rnd: int) -> None:
        t = self.now
        self.emitted_wave1 += int(wave == 1)
        self.emitted_wave2 += int(wave == 2)
        self.local_version[node] += 1
        msg = Message(next(self._mid), node, self.local_version[node], wave, rnd, t)
        self.capture_resources(msg)
        if self.rec is not None:
            self.rec.on_emit(t, msg)
        self.net.send(node, self.N, self.cfg.message_bytes + self.cfg.frame_overhead_bytes,
                      self.cfg.uplink_us * US, self._on_gcs_in, (msg,), kind="uplink")

    # ------------------------------------------------------------------ #
    # Event handlers
    # ------------------------------------------------------------------ #
    def _on_gcs_in(self, msg: Message) -> None:
        t = self.now
        msg.gcs_recv_t = t
        if msg.version <= self.gcs_versions[msg.src]:
            self.total_stale_dropped += 1
            if self.rec is not None:
                self.rec.on_gcs_in(t, msg, accepted=False, q_len=self.gcs_in_backlog)
            return
        self.gcs_versions[msg.src] = msg.version
        msg.gcs_in_q_at = self.gcs_in_backlog
        self.accepted += 1
        if self.rec is not None:
            self.rec.on_gcs_in(t, msg, accepted=True, q_len=self.gcs_in_backlog)
        self._submit_fanout(msg)

    def _on_exec_done(self, task: Tuple) -> None:
        self.exec_busy = False
        kind = task[0]
        if kind == E_FANOUT:
            msg = task[1]
            self.gcs_in_backlog -= 1
            msg.fanout_start_t = self.exec_start
            if self.rec is not None:
                self.rec.on_fanout(self.exec_start, self.now, msg)
            self._fanout(msg)
        else:
            _, dst, batch, start_t, trigger = task
            self._write_done(dst, batch, start_t, trigger)
        self._pump_exec()

    def _fanout(self, msg: Message) -> None:
        """RaySyncer::BroadcastMessage's loop over every connected reactor."""
        start = self.exec_start
        offset = (self.cfg.gcs_fixed_us + self.cfg.gcs_fanout_us) * US
        step = self.cfg.gcs_fanout_us * US
        for d in range(self.N):
            if d == msg.src:
                continue
            self.reactors[d].push(msg, start + offset + step * (d - 1), self)

    def _write_done(self, dst: int, batch: List[Tuple[Message, float, int]],
                    start_t: float, trigger: str) -> None:
        reactor = self.reactors[dst]
        reactor.inflight = None
        delivered = self.now + self.cfg.prop_us * US
        if self.rec is not None:
            self.rec.on_write(self.now, dst, batch, start_t, delivered, trigger)
        self.net.send(self.N, dst, self.cfg.message_bytes * len(batch) + self.cfg.frame_overhead_bytes,
                      self.cfg.prop_us * US, self._schedule_receive,
                      (dst, [item[0] for item in batch]), messages=len(batch), kind="propagation")
        # PR OnWriteDone -> SendNext -> StartSend: a partial pending buffer is
        # sent immediately, without waiting for another size/timeout trigger.
        reactor.pending_trigger = "write_done"
        reactor.start_write(self)

    def _on_batch_timer(self, node: int, seq: int) -> None:
        """``batch_timer_.async_wait`` callback of the bidi reactor."""
        r = self.reactors[node]
        if seq != r.timer_seq:
            return  # cancelled or superseded by a size-triggered flush
        r.batch_timer_active = False
        r.pending_trigger = "timer"
        r.start_write(self)

    def _schedule_receive(self, dst, messages):
        for msg in messages:
            start = max(self.now, self.node_proc_free[dst])
            wait = start - self.now
            self.processing_wait_s += wait
            self.processing_wait_max_s = max(self.processing_wait_max_s, wait)
            done = start + self.cfg.node_proc_us * US
            self.node_proc_free[dst] = done
            self._push(done, E_RECV, (dst, msg))

    def _on_recv(self, dst: int, msg: Message) -> None:
        t = self.now
        if self.first_recv[dst, msg.src] < 0:
            self.first_recv[dst, msg.src] = t
            self.w1_count[dst] += 1
        if msg.version > self.received_versions[dst, msg.src]:
            self.received_versions[dst, msg.src] = msg.version
            self.apply_resources(dst, msg)
            self.latest_version_recv[dst, msg.src] = t
        self.last_recv[dst, msg.src] = t
        self.reactors[dst].last_delivery_t = t
        if self.rec is not None:
            self.rec.on_recv(t, dst, msg)

    def _on_sample(self) -> None:
        if self.rec is None:
            return
        total = 0
        mx = 0
        for r in self.reactors:
            n = len(r.buffer)
            total += n
            if n > mx:
                mx = n
        converged = sum(1 for c in self.w1_count if c >= self.N - 1)
        trace = getattr(self.rec, "trace_nodes", ())
        trace_backlog = tuple(len(self.reactors[n].buffer) for n in trace)
        self.rec.on_sample(
            self.now,
            {
                "link_backlog_total": total,
                "link_backlog_max": mx,
                "gcs_in_backlog": self.gcs_in_backlog,
                "exec_queue": len(self.exec_q) + (1 if self.exec_busy else 0),
                "nodes_w1_converged": converged,
                "trace_backlog": trace_backlog,
            },
        )

    # ------------------------------------------------------------------ #
    # Driver
    # ------------------------------------------------------------------ #
    def run(self) -> SimResult:
        for s in range(1, int(self.cfg.max_sim_s) + 1):
            self._push_sample(float(s))

        self.start_wave1()

        heap = self.heap
        while heap and self.n_real > 0:
            if heap[0][0] > self.cfg.max_sim_s:
                self.now = self.cfg.max_sim_s
                self.termination_reason = "max_sim_s"
                break
            t, _, kind, payload = heapq.heappop(heap)
            if kind != E_SAMPLE:
                self.n_real -= 1
            self.now = t
            if kind == CALLBACK:
                fn, args = payload
                fn(*args)
            elif kind == E_GCS_IN:
                self._on_gcs_in(payload)
            elif kind == E_EXEC_DONE:
                self._on_exec_done(payload)
            elif kind == E_RECV:
                self._on_recv(*payload)
            elif kind == E_BATCH_TIMER:
                self._on_batch_timer(*payload)
            else:
                self._on_sample()
        self._on_sample()
        return self._finalize()

    def _finalize(self) -> SimResult:
        N = self.N
        eyes = np.arange(N)
        w1 = self.first_recv.copy()
        w1[eyes, eyes] = -1.0
        per_dst = np.array([r.writes for r in self.reactors], dtype=np.int64)
        per_dst_msgs = np.array([r.msgs_written for r in self.reactors],
                                dtype=np.int64)
        last_del = np.array([r.last_delivery_t for r in self.reactors], dtype=np.float64)

        batch_sizes: List[int] = []
        triggers: Dict[str, int] = {}
        for r in self.reactors:
            batch_sizes.extend(r.batch_sizes)
            for k, v in r.trigger_counts.items():
                triggers[k] = triggers.get(k, 0) + v

        pairs = ~np.eye(N, dtype=bool)
        wave1_conv = (float(w1[pairs].max())
                      if np.all(w1[pairs] >= 0) else float("nan"))
        final_versions = np.asarray(self.local_version, dtype=np.int64)
        covered = ((self.received_versions >= final_versions[None, :])
                   & (final_versions[None, :] >= 0) & pairs)
        uncovered = int(np.count_nonzero(pairs & ~covered))
        coverage = float(np.count_nonzero(covered) / (N * (N - 1)))
        # A time-limited run cannot certify completion of future feedback work.
        converged = (uncovered == 0 and self.termination_reason != "max_sim_s"
                     and (not hasattr(self, "business") or self.business.complete))
        final_conv = (float(self.latest_version_recv[pairs].max())
                      if converged else float("nan"))
        reason = ("converged" if converged else self.termination_reason
                  if self.termination_reason == "max_sim_s" else "incomplete")

        total_in = self.accepted + self.total_stale_dropped
        slots = N * (N - 1)

        return SimResult(
            cfg=self.cfg,
            num_nodes=N,
            wall_sim_time=self.now,
            total_link_writes=int(per_dst.sum()),
            total_messages_written=int(per_dst_msgs.sum()),
            total_inbound_messages=total_in,
            total_stale_dropped=self.total_stale_dropped,
            total_pushes=self.total_pushes,
            total_new_entry_writes=self.new_entry_writes,
            total_slot_writes=slots,
            total_resend_writes=self.new_entry_writes - slots,
            total_inplace_merges=self.inplace_merges,
            new_entries_by_src=self.new_entries_by_src,
            total_wave1_messages=self.emitted_wave1,
            total_wave2_messages=self.emitted_wave2,
            wave1_convergence_s=wave1_conv,
            final_convergence_s=final_conv,
            wave1_last_delivery_s=float(last_del.max()) if last_del.size else 0.0,
            per_dst_writes=per_dst,
            per_dst_messages=per_dst_msgs,
            last_delivery_per_dst=last_del,
            first_recv=self.first_recv,
            last_recv=self.last_recv,
            received_versions=self.received_versions,
            final_versions=final_versions.copy(),
            converged=converged,
            final_version_coverage=coverage,
            uncovered_final_pairs=uncovered,
            termination_reason=reason,
            batch_sizes=np.asarray(batch_sizes, dtype=np.int64),
            trigger_counts=triggers,
        )

