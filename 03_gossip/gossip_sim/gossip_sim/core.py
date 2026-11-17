"""Gossip over the common explicit-resource workload and finite NIC model.

Per-node CPU queues serialize relay, receive, and digest processing. AE digests
contain version snapshots and cross the network before comparison; request
frames return before repair data is sent. The origin holds its own version
immediately. Remote updates and their relays never create new local versions.
"""

from __future__ import annotations

import heapq
import itertools
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .params import US, GossipParams
from raysyncer_sim.costs import Network, ResourceWorkload, CALLBACK
from .topology import GossipTopology, pair_hash

MSG_RESOURCE_VIEW = 0

E_GCS_IN = "gcs_in"
E_EXEC = "exec"
E_RECV = "recv"
E_AE = "ae_round"
E_SAMPLE = "sample"
GCS = -1  # executor id of the GCS
RELAY_QUEUE = "latest_per_peer_source_two_stage_v2"
AE_QUEUE = "latest_per_peer_source_and_slice_v1"


class Message:
    __slots__ = ("resources", "mid", "src", "version", "wave", "rnd", "emit_t")

    def __init__(self, mid: int, src: int, version: int, wave: int, rnd: int,
                 emit_t: float):
        self.resources = None
        self.mid = mid
        self.src = src
        self.version = version
        self.wave = wave
        self.rnd = rnd
        self.emit_t = emit_t


@dataclass(frozen=True)
class RelaySnapshot:
    message: Message
    hop: int
    update_batch: int


@dataclass(frozen=True)
class ActiveSend:
    send_id: int
    snapshot: RelaySnapshot


@dataclass
class RelayUpdateWork:
    batch_id: int
    cost: float = 0.0
    updates: int = 0


@dataclass
class PeerRelayBuffer:
    pending: dict[int, RelaySnapshot] = field(default_factory=dict)
    scheduled: set[int] = field(default_factory=set)
    inflight: dict[int, ActiveSend] = field(default_factory=dict)


@dataclass(frozen=True)
class RepairSnapshot:
    message: Message
    extra_delay_s: float
    is_pull: bool


@dataclass(frozen=True)
class ActiveRepair:
    send_id: int
    snapshot: RepairSnapshot


@dataclass
class PeerRepairBuffer:
    pending: dict[int, RepairSnapshot] = field(default_factory=dict)
    scheduled: set[int] = field(default_factory=set)
    inflight: dict[int, ActiveRepair] = field(default_factory=dict)


@dataclass
class GossipResult:
    cfg: Any
    gparams: GossipParams
    num_nodes: int
    wall_sim_time: float
    topology: Dict[str, Any]
    # messages
    uplink_msgs: int            # node -> GCS attempts
    uplink_stale: int           # GCS-side stale drops
    uplink_accepted: int
    relay_msgs: int             # node -> node messages actually written
    relay_writes: int
    pushes_attempted: int
    pushes_suppressed_peer_has_it: int
    pushes_excluded: int
    # hop accounting
    hop_hist: np.ndarray
    max_hops: int
    mean_hops_at_delivery: float
    # final-version coverage: did every node learn every source's LAST version?
    final_version_coverage: float
    final_version_coverage_time_s: float
    uncovered_final_pairs: int
    # anti-entropy accounting
    ae_rounds: int
    ae_data_msgs: int
    ae_push_msgs: int
    ae_pull_msgs: int
    ae_digest_msgs: int
    ae_digest_bytes: int
    ae_all_milestones_reached_t: float
    ae_stopped_early: bool
    stale_repairs: int
    coverage_trace: list
    # receives
    received_total: int
    stale_received: int
    # per-node
    per_node_sent: np.ndarray
    per_node_received: np.ndarray
    max_node_sends: int
    max_node_sends_node: int
    # convergence / workload
    total_wave1_updates: int
    total_wave2_updates: int
    wave2_first_emit_t: Optional[float]
    wave1_convergence_s: float
    final_convergence_s: float
    first_recv: np.ndarray
    last_recv: np.ndarray
    emit_times: List[float] = field(default_factory=list)
    ae_data_sent: int = 0
    ae_repairs_suppressed: int = 0
    ae_pending_peak: int = 0
    ae_pending_at_end: int = 0
    ae_repair_overwrites: int = 0
    ae_requests_suppressed: int = 0
    ae_requests_pruned: int = 0
    ae_digest_overwrites: int = 0
    termination_reason: str = "event_queue_exhausted"
    relay_overwrites: int = 0
    relay_pending_peak: int = 0
    relay_inflight_peak: int = 0
    relay_pending_at_end: int = 0
    relay_inflight_at_end: int = 0
    relay_update_events: int = 0
    relay_update_events_completed: int = 0
    relay_update_cpu_completed_s: float = 0.0
    relay_write_cpu_completed_s: float = 0.0

    @property
    def total_messages(self) -> int:
        return self.uplink_msgs + self.relay_msgs


class GossipSimulator(ResourceWorkload):
    def __init__(self, cfg, gparams: GossipParams, recorder=None):
        self.cfg = cfg
        self.gp = gparams
        self.rec = recorder
        self.N = int(cfg.num_nodes)
        self.net = Network(self)
        self.resource_state = [0] * self.N
        self.emitted_wave1 = 0
        self.emitted_wave2 = 0
        self.executor_wait_s = 0.0
        self.executor_wait_max_s = 0.0
        self.rng = random.Random(cfg.seed)
        self.topology = GossipTopology(self.N, gparams.degree_k,
                                       gparams.topology_seed,
                                       neighbor_mode=gparams.neighbor_mode)
        # per-node RNGs for the "random" fan-out mode (kept separate from the
        # workload RNG so the generator stays identical across arms)
        self.fanout_rngs = [
            random.Random(f"{gparams.topology_seed}:fanout:{i}")
            for i in range(self.N)
        ]
        self.hop = gparams.resolved_hop_us(cfg) * US

        self.heap: List[Tuple[float, int, str, Any]] = []
        self._seq = itertools.count()
        self._mid = itertools.count()
        self._send_id = itertools.count()
        self.now = 0.0

        # --- per-node executors (each Raylet's own io_context) ------------- #
        self.exec_q: list[deque[tuple[tuple, float, float]]] = [
            deque() for _ in range(self.N + 1)  # +1 slot for the GCS
        ]
        self.exec_busy = [False] * (self.N + 1)
        self.exec_start = [0.0] * (self.N + 1)

        def slot(node: int) -> int:
            return 0 if node == GCS else node + 1

        self._slot = slot

        # --- state that mirrors the C++ reactors --------------------------- #
        # local view per node: highest version accepted per (src)
        self.view_version = np.full((self.N, self.N), -1, dtype=np.int64)
        # Per-edge version watermark, advanced when the sending NIC finishes.
        self.completed_version = [
            np.full((len(self.topology.out[i]), self.N), -1, dtype=np.int64)
            for i in range(self.N)
        ]
        self.peer_slot = [
            {peer: s for s, peer in enumerate(self.topology.out[i])}
            for i in range(self.N)
        ]
        self.relay_buffers = [
            {peer: PeerRelayBuffer() for peer in peers}
            for peers in self.topology.out
        ]
        self.relay_pending_count = 0
        self.relay_inflight_count = 0
        self.relay_overwrites = 0
        self.relay_pending_peak = 0
        self.relay_inflight_peak = 0
        # At most one queued first-stage token per node. At service start,
        # detach its accumulated work so later arrivals belong to a new batch.
        self.relay_update_work: list[Optional[RelayUpdateWork]] = [None] * self.N
        self.relay_update_batch = [0] * self.N
        self.relay_processed_batch = [0] * self.N
        self.relay_update_events = 0
        self.relay_update_events_completed = 0
        self.relay_update_cpu_completed_s = 0.0
        self.relay_write_cpu_completed_s = 0.0

        # --- workload (identical to stage 1) ------------------------------- #
        self.local_version = [0] * self.N
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

        # per-(sender, target) write cost, same jitter model as stage 1
        j = cfg.link_write_jitter
        self.write_cost = [
            [
                cfg.gcs_write_us * US
                * (1.0 + (j * (self.rng.random() - 0.5) if j > 0 else 0.0))
                for _ in self.topology.out[i]
            ]
            for i in range(self.N)
        ]

        # --- bookkeeping --------------------------------------------------- #
        self.first_recv = np.full((self.N, self.N), -1.0)
        self.last_recv = np.full((self.N, self.N), -1.0)
        self.last_accept_t = np.full((self.N, self.N), -1.0)
        self.per_node_sent = np.zeros(self.N, dtype=np.int64)
        self.per_node_received = np.zeros(self.N, dtype=np.int64)

        self.uplink_msgs = 0
        self.uplink_stale = 0
        self.uplink_accepted = 0
        self.relay_msgs = 0
        self.relay_writes = 0
        self.pushes_attempted = 0
        self.pushes_suppressed = 0
        self.pushes_excluded = 0
        self.received_total = 0
        self.stale_received = 0
        self.wave2_first_emit_t: Optional[float] = None
        self.emit_times: List[float] = []
        self.gcs_versions: Dict[int, int] = {}
        self.hop_hist = np.zeros(64, dtype=np.int64)
        self.hop_sum = 0
        # anti-entropy state
        self.ae_rounds = 0
        self.ae_data_msgs = 0
        self.ae_push_msgs = 0
        self.ae_pull_msgs = 0
        self.ae_digest_msgs = 0
        self.ae_digest_bytes = 0
        self.ae_all_milestones_t = -1.0
        self.ae_stopped_early = False
        self._ae_events = 0
        self.ae_deadline_s = gparams.resolved_ae_deadline_s(cfg)
        self.ae_repair_buffers: list[dict[int, PeerRepairBuffer]] = [
            {} for _ in range(self.N)
        ]
        # Frozen repairs remain here until receive processing, beyond tx completion.
        self.ae_pending: set[tuple[int, int, int, int]] = set()
        self.ae_requests: dict[tuple[int, int, int], set[int]] = {}
        self.ae_digest_pending: dict[tuple[int, int, int, bool], list[int] | tuple[int, ...]] = {}
        self.ae_repairs_suppressed = 0
        self.ae_pending_peak = 0
        self.ae_repair_overwrites = 0
        self.ae_requests_suppressed = 0
        self.ae_requests_pruned = 0
        self.ae_digest_overwrites = 0
        self.ae_data_sent = 0
        self.termination_reason = "event_queue_exhausted"
        # how many destinations currently hold src's *latest* version
        self.have_final_counts = np.zeros(self.N, dtype=np.int64)
        self.stale_repairs = 0
        self.coverage_trace: List[Tuple[float, float]] = []
        # pre-computed digest slices: ae_slices[node][slice_idx] = list of srcs
        self.ae_slices: List[List[List[int]]] = []
        if gparams.antientropy_period_ms > 0:
            S = max(1, gparams.antientropy_slices)
            for node in range(self.N):
                buckets: List[List[int]] = [[] for _ in range(S)]
                for src in range(self.N):
                    buckets[pair_hash(src, node, gparams.topology_seed) % S].append(src)
                self.ae_slices.append(buckets)

    # ------------------------------------------------------------------ #
    # event plumbing
    # ------------------------------------------------------------------ #
    def _push(self, t: float, kind: str, payload: Any) -> None:
        heapq.heappush(self.heap, (t, next(self._seq), kind, payload))

    def _submit(self, node: int, task: Tuple, cost: float) -> None:
        s = self._slot(node)
        self.exec_q[s].append((task, cost, self.now))
        self._pump(node)

    def _pump(self, node: int) -> None:
        s = self._slot(node)
        if self.exec_busy[s] or not self.exec_q[s]:
            return
        # Keep blocked slot tokens in their FIFO positions. A newer snapshot
        # may still need first-stage CPU work queued behind an existing token.
        queue = self.exec_q[s]
        while True:
            for index, (task, cost, queued) in enumerate(queue):
                if task[0] != "relay_slot":
                    break
                _, peer, src = task
                snapshot = self.relay_buffers[node][peer].pending[src]
                if snapshot.update_batch <= self.relay_processed_batch[node]:
                    break
            else:
                return
            del queue[index]
            if task[0] in ("ae_digest_slot", "ae_request"):
                task, cost = self._prepare_ae_control(node, task)
                if task is None:
                    continue
            break
        wait = self.now - queued
        self.executor_wait_s += wait
        self.executor_wait_max_s = max(self.executor_wait_max_s, wait)
        self.exec_busy[s] = True
        self.exec_start[s] = self.now
        if task[0] == "relay_update":
            work = self.relay_update_work[node]
            self.relay_update_work[node] = None
            task = ("relay_update_done", work)
            cost = work.cost
        elif task[0] == "relay_slot":
            _, peer, src = task
            active = self._start_relay_slot(node, peer, src)
            cost = (self._link_write_cost(node, peer)
                    + (self.gp.resolved_msg_bytes() + self.gp.frame_overhead_bytes)
                    * self.cfg.serialize_us_per_byte * US)
            task = ("relay_write", peer, active, cost)
        elif task[0] == "ae_repair_slot":
            _, peer, src = task
            task = ("ae_repair_write", peer, self._start_ae_repair(node, peer, src))
        self._push(self.now + cost, E_EXEC, (node, task))

    # ------------------------------------------------------------------ #
    # workload (identical semantics to stage 1)
    # ------------------------------------------------------------------ #
    def emit(self, node: int, wave: int, rnd: int) -> None:
        t = self.now
        self.emitted_wave1 += int(wave == 1)
        self.emitted_wave2 += int(wave == 2)
        self.local_version[node] += 1
        msg = Message(next(self._mid), node, self.local_version[node], wave, rnd, t)
        self.capture_resources(msg)
        # a fresh version resets the "who holds the latest" counter for its source
        self.have_final_counts[node] = 0
        self.view_version[node, node] = msg.version
        if wave == 2 and self.wave2_first_emit_t is None:
            self.wave2_first_emit_t = t
        self.emit_times.append(t)
        if self.rec is not None:
            self.rec.on_emit(t, msg)
        # (a) uplink to the GCS: same as the star arm; the GCS no longer fans out
        self.net.send(node, self.N, self.cfg.message_bytes + self.cfg.frame_overhead_bytes,
                      self.cfg.uplink_us * US, self._on_gcs_in, (msg,), kind="uplink")
        # (b) the origin relays to f neighbours on its own executor
        self._schedule_broadcast(node, msg, exclude=())

    # ------------------------------------------------------------------ #
    # propagation
    # ------------------------------------------------------------------ #
    def _schedule_broadcast(self, node: int, msg: Message,
                            exclude: Tuple[int, ...], hop: int = 0) -> None:
        targets = self.topology.select_fanout(
            node, msg.mid, msg.src, exclude, self.gp.fanout_f,
            mode=self.gp.fanout_mode, rng=self.fanout_rngs[node])
        work = self.relay_update_work[node]
        if work is None:
            self.relay_update_batch[node] += 1
            work = RelayUpdateWork(self.relay_update_batch[node])
            self.relay_update_work[node] = work
            self.exec_q[self._slot(node)].append((("relay_update",), 0.0, self.now))
        # First-stage processing is paid once per update, even if its snapshot
        # is overwritten or all selected targets suppress it as stale.
        work.cost += (self.cfg.gcs_fixed_us + len(targets) * self.cfg.gcs_fanout_us) * US
        work.updates += 1
        self.relay_update_events += 1
        if not targets:
            self.pushes_excluded += 1
        for peer in targets:
            self._offer_relay(node, peer, msg, hop, work.batch_id)
        self._pump(node)

    def _offer_relay(self, node: int, peer: int, msg: Message, hop: int,
                     update_batch: int) -> None:
        self.pushes_attempted += 1
        src = msg.src
        buffer = self.relay_buffers[node][peer]
        pending = buffer.pending.get(src)
        active = buffer.inflight.get(src)
        slot = self.peer_slot[node][peer]
        if msg.version <= max(
            self.completed_version[node][slot, src],
            pending.message.version if pending else -1,
            active.snapshot.message.version if active else -1,
        ):
            self.pushes_suppressed += 1
            return

        if pending is None:
            self.relay_pending_count += 1
            self.relay_pending_peak = max(self.relay_pending_peak, self.relay_pending_count)
        else:
            self.relay_overwrites += 1
        buffer.pending[src] = RelaySnapshot(msg, hop, update_batch)
        self._schedule_relay_slot(node, peer, src)

    def _schedule_relay_slot(self, node: int, peer: int, src: int) -> None:
        buffer = self.relay_buffers[node][peer]
        if src in buffer.pending and src not in buffer.scheduled and src not in buffer.inflight:
            buffer.scheduled.add(src)
            self.exec_q[self._slot(node)].append((("relay_slot", peer, src), 0.0, self.now))

    def _start_relay_slot(self, node: int, peer: int, src: int) -> ActiveSend:
        buffer = self.relay_buffers[node][peer]
        buffer.scheduled.remove(src)
        snapshot = buffer.pending.pop(src)
        active = ActiveSend(next(self._send_id), snapshot)
        buffer.inflight[src] = active
        self.relay_pending_count -= 1
        self.relay_inflight_count += 1
        self.relay_inflight_peak = max(self.relay_inflight_peak, self.relay_inflight_count)
        return active

    def _finish_relay_slot(self, node: int, peer: int, src: int, send_id: int) -> None:
        buffer = self.relay_buffers[node][peer]
        active = buffer.inflight.get(src)
        if active is None or active.send_id != send_id:
            return
        slot = self.peer_slot[node][peer]
        self.completed_version[node][slot, src] = active.snapshot.message.version
        del buffer.inflight[src]
        self.relay_inflight_count -= 1
        self._schedule_relay_slot(node, peer, src)
        self._pump(node)

    def _do_write(self, node: int, msg: Message, dst: int, hop: int,
                   relay: bool = True, extra_delay_s: float = 0.0,
                   *, on_tx_done=None) -> None:
        if relay:
            self.relay_writes += 1
            self.relay_msgs += 1
        else:
            self.ae_data_sent += 1
        self.per_node_sent[node] += 1
        size = self.gp.resolved_msg_bytes() + self.gp.frame_overhead_bytes
        self.net.send(node, dst, size, self.hop + extra_delay_s, self._queue_receive,
                      (dst, msg, node, hop + 1, relay),
                      kind="relay" if relay else "ae_data", on_tx_done=on_tx_done)

    # ------------------------------------------------------------------ #
    # anti-entropy (design §6.2): periodic sliced digest + targeted repair
    # ------------------------------------------------------------------ #
    def _start_antientropy(self) -> None:
        p = self.gp.antientropy_period_ms * 1e-3
        if p <= 0:
            return
        # stagger the nodes so the digest rounds do not fire in lockstep
        for node in range(self.N):
            t0 = p * (1.0 + node / max(1, self.N))
            if t0 <= self.ae_deadline_s:
                self._push(t0, E_AE, (node, 0))

    def _ae_slice_sources(self, node: int, slice_idx: int):
        S = max(1, self.gp.antientropy_slices)
        return self.ae_slices[node][slice_idx % S]

    def view_of(self, node: int, src: int) -> int:
        """The version of ``src`` held by ``node``.

        A node always holds its own latest version (``NodeState::cluster_view_``
        is populated by ``BroadcastMessage`` for the local reporter too), which
        the replica matrix does not store on the diagonal.
        """
        if node == src:
            return int(self.local_version[src])
        return int(self.view_version[node, src])

    def _on_ae_round(self, node: int, r: int) -> None:
        if self.ae_stopped_early:
            return
        self.ae_rounds += 1
        outs = self.topology.out[node]
        slice_idx = r % max(1, self.gp.antientropy_slices)
        srcs = self._ae_slice_sources(node, slice_idx)
        n_peers = max(1, min(self.gp.antientropy_peers, len(outs)))
        pushpull = self.gp.ae_mode == "pushpull"
        for p in range(n_peers):
            peer = outs[(r * 7919 + node + p * 131) % len(outs)]
            self._send_digest(node, peer, slice_idx, srcs, pushpull)
        # stop once everyone holds everyone's latest version and no new
        # milestones can fire (all nodes saw the whole cluster)
        if self.now >= self.workload_last_s and self.have_final_counts.min() >= self.N - 1:
            self.ae_stopped_early = True
            self.ae_all_milestones_t = self.now
            return
        if self.now + self.gp.antientropy_period_ms * 1e-3 <= self.ae_deadline_s:
            self._push(self.now + self.gp.antientropy_period_ms * 1e-3,
                       E_AE, (node, r + 1))

    def _send_digest(self, node, peer, slice_idx, srcs, reply):
        # A pushpull response refers to the initiating peer's partition.
        key = (node, peer, slice_idx, reply)
        already_queued = key in self.ae_digest_pending
        self.ae_digest_pending[key] = srcs
        if already_queued:
            self.ae_digest_overwrites += 1
        else:
            self._submit(node, ("ae_digest_slot", key), 0.0)

    def _prepare_ae_control(self, node, task):
        if task[0] == "ae_digest_slot":
            key = task[1]
            _, peer, slice_idx, reply = key
            srcs = self.ae_digest_pending.pop(key)
            entries = tuple((src, self.view_of(node, src)) for src in srcs)
        else:
            _, peer, requests = task
            slice_idx, reply = None, "request"
            needed = []
            for src, version in requests:
                if version > self.view_of(node, src):
                    needed.append((src, version))
                else:
                    self.ae_requests_pruned += 1
                    self._finish_ae_requests(node, peer, src, version)
            if not needed:
                return None, 0.0
            entries = tuple(needed)
        size = (self.gp.digest_header_bytes + self.gp.digest_entry_bytes * len(entries)
                + self.gp.frame_overhead_bytes)
        cost = self._link_write_cost(node, peer) + size * self.cfg.serialize_us_per_byte * US
        return ("digest_write", node, peer, entries, reply, size, slice_idx), cost

    def _digest_write(self, node, peer, entries, reply, size, slice_idx):
        self.ae_digest_msgs += 1
        self.ae_digest_bytes += size - self.gp.frame_overhead_bytes
        self.net.send(node, peer, size, self.hop, self._queue_digest,
                      (node, peer, entries, reply, slice_idx), kind="digest")

    def _queue_digest(self, node, peer, entries, reply, slice_idx):
        self._submit(peer, ("digest_receive", node, peer, entries, reply, slice_idx),
                     self.cfg.gcs_fixed_us * US + len(entries) * self.cfg.gcs_fanout_us * US)

    def _digest_received(self, node, peer, entries, reply, slice_idx):
        if reply == "request":
            for src, version in entries:
                if self.view_of(peer, src) >= version:
                    self._ae_send(peer, node, src, version)
                else:
                    self._finish_ae_requests(node, peer, src, version, exact=True)
            return
        requests = []
        for src, version in entries:
            if version <= 0 or version <= self.view_of(peer, src):
                continue
            versions = self.ae_requests.setdefault((peer, node, src), set())
            if version in versions:
                self.ae_requests_suppressed += 1
                continue
            versions.add(version)
            requests.append((src, version))
        if requests:
            self._submit(peer, ("ae_request", node, tuple(requests)), 0.0)
        if reply is True:
            self._send_digest(peer, node, slice_idx, tuple(src for src, _ in entries), False)

    def _finish_ae_requests(self, requester, provider, src, version, *, exact=False):
        key = (requester, provider, src)
        versions = self.ae_requests.get(key)
        if versions is not None:
            if exact:
                versions.discard(version)
            else:
                versions.difference_update(v for v in tuple(versions) if v <= version)
            if not versions:
                del self.ae_requests[key]

    def _ae_send(self, node: int, peer: int, src: int, version: int,
                 delay_extra_s: float = 0.0, is_pull: bool = False) -> None:
        if hasattr(self, "replica_resources"):
            # Reply with the sender's own held snapshot; never read a source's future state.
            held = self.view_of(node, src)
            if held < version:
                raise RuntimeError("repair sender does not hold the requested version")
            version = held
        peers = self.ae_repair_buffers[node]
        buffer = peers.get(peer)
        if buffer is None:
            buffer = peers[peer] = PeerRepairBuffer()
        pending = buffer.pending.get(src)
        active = buffer.inflight.get(src)
        if ((pending is not None and version <= pending.message.version)
                or (active is not None and version < active.snapshot.message.version)
                or (self.gp.ae_deduplicate_inflight
                    and (node, peer, src, version) in self.ae_pending)):
            self.ae_repairs_suppressed += 1
            return
        msg = Message(next(self._mid), src, version, wave=0, rnd=0,
                      emit_t=self.now)
        msg.resources = (self.replica_resources[node, src] if hasattr(self, "replica_resources")
                         else getattr(self, "resource_history", {}).get((src, version)))
        self.ae_data_msgs += 1
        if pending is not None:
            self.ae_repair_overwrites += 1
        buffer.pending[src] = RepairSnapshot(msg, delay_extra_s, is_pull)
        self._schedule_ae_repair(node, peer, src)

    def _schedule_ae_repair(self, node, peer, src):
        buffer = self.ae_repair_buffers[node][peer]
        if src in buffer.pending and src not in buffer.scheduled and src not in buffer.inflight:
            buffer.scheduled.add(src)
            size = self.gp.resolved_msg_bytes() + self.gp.frame_overhead_bytes
            cost = self._link_write_cost(node, peer) + size * self.cfg.serialize_us_per_byte * US
            self._submit(node, ("ae_repair_slot", peer, src), cost)

    def _start_ae_repair(self, node, peer, src):
        buffer = self.ae_repair_buffers[node][peer]
        buffer.scheduled.remove(src)
        active = ActiveRepair(next(self._send_id), buffer.pending.pop(src))
        buffer.inflight[src] = active
        if self.gp.ae_deduplicate_inflight:
            self.ae_pending.add((node, peer, src, active.snapshot.message.version))
            self.ae_pending_peak = max(self.ae_pending_peak, len(self.ae_pending))
        return active

    def _finish_ae_repair(self, node, peer, src, send_id):
        buffer = self.ae_repair_buffers[node][peer]
        active = buffer.inflight.get(src)
        if active is None or active.send_id != send_id:
            return
        del buffer.inflight[src]
        self._schedule_ae_repair(node, peer, src)

    def _link_write_cost(self, node: int, peer: int) -> float:
        slot = self.peer_slot[node].get(peer)
        if slot is not None:
            return self.write_cost[node][slot]
        # reverse direction of a link this node did not dial: same cost model,
        # deterministic per pair.
        j = self.cfg.link_write_jitter
        h = pair_hash(node, peer, self.gp.topology_seed) / float(1 << 64)
        return self.cfg.gcs_write_us * US * (1.0 + j * (h - 0.5))

    # ------------------------------------------------------------------ #
    # handlers
    # ------------------------------------------------------------------ #
    def _on_gcs_in(self, msg: Message) -> None:
        self.uplink_msgs += 1
        if msg.version <= self.gcs_versions.get(msg.src, -1):
            self.uplink_stale += 1
            return
        self.gcs_versions[msg.src] = msg.version
        self.uplink_accepted += 1
        # gossip: the GCS consumes the update (its authoritative view) but does
        # not fan it out -- that is the whole point of the protocol change.
        self._submit(GCS, ("gcs_consume", msg), self.cfg.gcs_fixed_us * US)

    def _on_exec_done(self, node: int, task: Tuple) -> None:
        s = self._slot(node)
        kind = task[0]
        if kind == "relay_update_done":
            work = task[1]
            self.relay_processed_batch[node] = work.batch_id
            self.relay_update_events_completed += work.updates
            self.relay_update_cpu_completed_s += work.cost
        elif kind == "relay_write":
            _, peer, active, cost = task
            snapshot = active.snapshot
            self.relay_write_cpu_completed_s += cost
            self._do_write(node, snapshot.message, peer, snapshot.hop,
                           on_tx_done=lambda: self._finish_relay_slot(
                               node, peer, snapshot.message.src, active.send_id))
        elif kind == "ae_repair_write":
            _, peer, active = task
            snapshot = active.snapshot
            if snapshot.is_pull:
                self.ae_pull_msgs += 1
            else:
                self.ae_push_msgs += 1
            self._do_write(node, snapshot.message, peer, 0, relay=False,
                           extra_delay_s=snapshot.extra_delay_s,
                           on_tx_done=lambda: self._finish_ae_repair(
                               node, peer, snapshot.message.src, active.send_id))
        elif kind == "write":
            self._do_write(node, task[1], task[2], task[3], task[4], task[5])
        elif kind == "receive":
            self._on_recv(*task[1:])
        elif kind == "digest_receive":
            self._digest_received(*task[1:])
        elif kind == "digest_write":
            self._digest_write(*task[1:])
        # "gcs_consume" has no further work
        # A receive can update multiple relay slots. Pump only after all
        # selected targets have been updated, before freezing the next write.
        self.exec_busy[s] = False
        self._pump(node)

    def _queue_receive(self, dst, msg, sender, hop, fresh):
        self._submit(dst, ("receive", dst, msg, sender, hop, fresh),
                     self.cfg.node_proc_us * US)

    def _on_recv(self, dst: int, msg: Message, sender: int, hop: int = 1,
                 fresh: bool = True) -> None:
        """``fresh`` marks a first-hand propagation of a new update; anti-entropy
        repairs are delivered with ``fresh=False`` so they stay out of the
        propagation-latency statistics.  Every accepted message is relayed, like
        ``RaySyncer::BroadcastMessage`` does in C++."""
        t = self.now
        if not fresh:
            self.ae_pending.discard((sender, dst, msg.src, msg.version))
            self._finish_ae_requests(dst, sender, msg.src, msg.version)
        self.received_total += 1
        self.per_node_received[dst] += 1
        if fresh:
            self.hop_hist[min(hop, len(self.hop_hist) - 1)] += 1
            self.hop_sum += hop
        if msg.version <= self.view_version[dst, msg.src]:
            self.stale_received += 1
            if (self.gp.repair_on_stale
                    and msg.version < self.view_of(dst, msg.src)):
                # the sender lags: send it what we have instead of dropping
                self.stale_repairs += 1
                self._ae_send(dst, sender, msg.src,
                              self.view_of(dst, msg.src))
            return
        self.view_version[dst, msg.src] = msg.version
        self.apply_resources(dst, msg)
        self.last_accept_t[dst, msg.src] = t
        if msg.version == self.local_version[msg.src]:
            self.have_final_counts[msg.src] += 1
        if self.first_recv[dst, msg.src] < 0:
            self.first_recv[dst, msg.src] = t
            self.w1_count[dst] += 1
        self.last_recv[dst, msg.src] = t
        if self.rec is not None and fresh:
            self.rec.on_recv(t, dst, msg)
        # relay onward (only because the local view actually advanced)
        exclude = (sender,) if self.gp.exclude_last_hop else ()
        self._schedule_broadcast(dst, msg, exclude=exclude, hop=hop)

    # ------------------------------------------------------------------ #
    def run(self) -> GossipResult:
        self.start_wave1()
        self._start_antientropy()
        if self.gp.antientropy_period_ms > 0 or self.gp.repair_on_stale:
            self._push(0.0, E_SAMPLE, None)
        heap = self.heap
        while heap:
            if heap[0][0] > self.cfg.max_sim_s:
                self.now = self.cfg.max_sim_s
                self.termination_reason = "max_sim_s"
                break
            t, _, kind, payload = heapq.heappop(heap)
            self.now = t
            if kind == CALLBACK:
                fn, args = payload
                fn(*args)
            elif kind == E_EXEC:
                self._on_exec_done(*payload)
            elif kind == E_RECV:
                self._on_recv(*payload)
            elif kind == E_GCS_IN:
                self._on_gcs_in(payload)
            elif kind == E_AE:
                self._on_ae_round(*payload)
            elif kind == E_SAMPLE:
                self._on_sample()
        return self._finalize()

    def _on_sample(self) -> None:
        pairs = max(1, self.N * (self.N - 1))
        cov = float(min(self.have_final_counts.sum(), pairs) / pairs)
        self.coverage_trace.append((round(self.now, 4), round(cov, 5)))
        nxt = self.now + 0.1
        if nxt <= self.ae_deadline_s and not self.ae_stopped_early:
            self._push(nxt, E_SAMPLE, None)

    def _finalize(self) -> GossipResult:
        N = self.N
        eyes = np.arange(N)
        w1 = self.first_recv.copy()
        fin = self.last_recv.copy()
        w1[eyes, eyes] = -1.0
        fin[eyes, eyes] = -1.0
        wave1_conv = float(w1.max()) if w1.size else 0.0
        max_node = int(self.per_node_sent.max()) if N else 0
        final_v = np.asarray(self.local_version, dtype=np.int64)
        cov = (self.view_version == final_v[None, :])
        cov[eyes, eyes] = False
        pairs = N * (N - 1)
        uncovered = int(pairs - cov.sum()) if pairs else 0
        cov_t = float(self.last_accept_t[cov].max()) if cov.any() else -1.0
        final_conv = (cov_t if uncovered == 0 and self.termination_reason != "max_sim_s"
                      and (not hasattr(self, "business") or self.business.complete) else float("nan"))
        return GossipResult(
            cfg=self.cfg,
            gparams=self.gp,
            num_nodes=N,
            wall_sim_time=self.now,
            topology=self.topology.summary(),
            uplink_msgs=self.uplink_msgs,
            uplink_stale=self.uplink_stale,
            uplink_accepted=self.uplink_accepted,
            relay_msgs=self.relay_msgs,
            relay_writes=self.relay_writes,
            pushes_attempted=self.pushes_attempted,
            pushes_suppressed_peer_has_it=self.pushes_suppressed,
            pushes_excluded=self.pushes_excluded,
            hop_hist=self.hop_hist,
            max_hops=int(np.nonzero(self.hop_hist)[0].max()) if self.hop_sum else 0,
            mean_hops_at_delivery=(self.hop_sum / self.received_total
                                   if self.received_total else 0.0),
            final_version_coverage=(cov.sum() / pairs) if pairs else 1.0,
            final_version_coverage_time_s=cov_t,
            uncovered_final_pairs=uncovered,
            ae_rounds=self.ae_rounds,
            ae_data_msgs=self.ae_data_msgs,
            ae_push_msgs=self.ae_push_msgs,
            ae_pull_msgs=self.ae_pull_msgs,
            ae_digest_msgs=self.ae_digest_msgs,
            ae_digest_bytes=self.ae_digest_bytes,
            ae_all_milestones_reached_t=self.ae_all_milestones_t,
            ae_stopped_early=self.ae_stopped_early,
            stale_repairs=self.stale_repairs,
            coverage_trace=self.coverage_trace,
            received_total=self.received_total,
            stale_received=self.stale_received,
            per_node_sent=self.per_node_sent,
            per_node_received=self.per_node_received,
            max_node_sends=max_node,
            max_node_sends_node=int(self.per_node_sent.argmax()) if N else -1,
            total_wave1_updates=self.emitted_wave1,
            total_wave2_updates=self.emitted_wave2,
            wave2_first_emit_t=self.wave2_first_emit_t,
            wave1_convergence_s=wave1_conv,
            final_convergence_s=final_conv,
            first_recv=self.first_recv,
            last_recv=self.last_recv,
            emit_times=self.emit_times,
            ae_data_sent=self.ae_data_sent,
            ae_repairs_suppressed=self.ae_repairs_suppressed,
            ae_pending_peak=self.ae_pending_peak,
            ae_pending_at_end=len(self.ae_pending),
            ae_repair_overwrites=self.ae_repair_overwrites,
            ae_requests_suppressed=self.ae_requests_suppressed,
            ae_requests_pruned=self.ae_requests_pruned,
            ae_digest_overwrites=self.ae_digest_overwrites,
            termination_reason=self.termination_reason,
            relay_overwrites=self.relay_overwrites,
            relay_pending_peak=self.relay_pending_peak,
            relay_inflight_peak=self.relay_inflight_peak,
            relay_pending_at_end=self.relay_pending_count,
            relay_inflight_at_end=self.relay_inflight_count,
            relay_update_events=self.relay_update_events,
            relay_update_events_completed=self.relay_update_events_completed,
            relay_update_cpu_completed_s=self.relay_update_cpu_completed_s,
            relay_write_cpu_completed_s=self.relay_write_cpu_completed_s,
        )
