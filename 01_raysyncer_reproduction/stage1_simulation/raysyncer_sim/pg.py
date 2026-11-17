"""Source-grounded PG workload on the existing Original Push event kernel.

Reference: ray-project/ray PR #57641, head d42cf042, base aaeb5f8f.
Python event/object names below are model abstractions, not upstream classes.
Only the fixed-node STRICT_PACK, CPU/node bundles and ready probe are supported.
There are no milestone thresholds or prescribed wave-2 start times.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
import heapq
import math
from typing import Optional

import numpy as np

from .config import SimConfig, US
from .costs import CALLBACK
from .core import E_EXEC_DONE, E_FANOUT, E_GCS_IN, E_RECV, E_SAMPLE, Message, Simulator

EPS = 1e-8
READY_AMOUNT = 0.001
BUNDLE_CAPACITY = 1000.0


@dataclass
class PGConfig(SimConfig):
    """Timing values are explicit modelling assumptions, not Ray measurements."""
    cpus_per_node: float = 16.0
    node_resource_capacity: float = 1.0
    bundles_per_pg: int = 8
    cpus_per_bundle: float = 1.0
    node_per_bundle: float = 0.01
    skip_head: bool = True
    report_period_s: float = 0.1  # Ray default, configurable in the real runtime.
    stagger_report_phase: bool = True
    control_oneway_s: float = 0.0005
    gcs_control_us: float = 10.0
    bundle_op_us: float = 10.0
    schedule_us: float = 20.0
    ready_runtime_s: float = 0.001
    lease_hold_s: float = 0.005
    wait_before_ready_s: float = 0.0
    submit_ready: bool = True
    scheduler_view: str = "received"  # oracle is a diagnostic control only.
    max_pg_attempts: int = 3
    retry_backoff_s: float = 0.1
    audit_resource_apply: bool = True

    def __post_init__(self):
        super().__post_init__()
        if not 0 <= self.driver_node < self.num_nodes:
            raise ValueError("invalid driver_node")
        if self.bundles_per_pg < 1 or self.max_pg_attempts < 1:
            raise ValueError("bundle count and attempts must be positive")
        if self.cpus_per_bundle <= 0 or self.node_per_bundle <= 0:
            raise ValueError("bundle resource requests must be positive")
        if self.cpus_per_node < 0 or self.node_resource_capacity < 0:
            raise ValueError("negative capacity")
        if self.report_period_s <= 0 or self.scheduler_view not in ("received", "oracle"):
            raise ValueError("invalid reporter period or scheduler view")
        for key in ("control_oneway_s", "gcs_control_us", "bundle_op_us", "schedule_us",
                    "ready_runtime_s", "lease_hold_s", "wait_before_ready_s", "node_proc_us"):
            if getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative")


@dataclass
class BundleRecord:
    index: int
    cpu: float
    node: float
    state: str = "PENDING"


@dataclass
class PGRecord:
    pg_id: int
    owner: int
    bundles: list[BundleRecord]
    strategy: str = "STRICT_PACK"
    state: str = "PENDING"
    attempts: int = 0
    registered_s: Optional[float] = None
    created_s: Optional[float] = None
    failed_s: Optional[float] = None


@dataclass
class ReadyTask:
    pg_id: int
    status: str = "PENDING"
    submitted_s: float = 0.0
    granted_s: Optional[float] = None
    done_s: Optional[float] = None
    released_s: Optional[float] = None
    decision_version: Optional[int] = None
    decision_bundle_available: Optional[float] = None
    attempts: int = 0


@dataclass(frozen=True)
class ResourceSnapshot:
    src: int
    version: int
    total: tuple[tuple[str, float], ...]
    available: tuple[tuple[str, float], ...]
    causes: tuple[str, ...] = ()

    def available_amount(self, key):
        return next((v for k, v in self.available if k == key), 0.0)


class PGMessage(Message):
    __slots__ = ("snapshot",)

    def __init__(self, mid, snapshot, t):
        # wave/rnd are legacy recorder labels only; they never trigger work.
        phase = 2 if any(c.startswith("lease_") for c in snapshot.causes) else 1
        super().__init__(mid, snapshot.src, snapshot.version, phase, 0, t)
        self.snapshot = snapshot


@dataclass
class NodeResources:
    capacity_cpu: float
    capacity_node: float
    free_cpu: float
    free_node: float
    reserved_cpu: float = 0.0
    reserved_node: float = 0.0
    pg_total: dict[str, float] = field(default_factory=dict)
    pg_available: dict[str, float] = field(default_factory=dict)


class PGRecorder:
    """GCS ingress, recipient writes, applies and business events separately."""
    trace_nodes = ()

    def __init__(self, origin):
        self.origin = origin
        self.bins = defaultdict(Counter)
        self.trace = []
        self.latencies = defaultdict(list)
        self.all_queues_peak = 0
        self.single_queue_peak = 0

    def count(self, t, name, n=1):
        self.bins[max(0, math.floor(t - self.origin + 1e-10))][name] += n

    def event(self, t, kind, pg_id=None, node=None, **extra):
        self.trace.append(dict(time_s=t-self.origin, event=kind, pg_id=pg_id, node=node, **extra))
        self.count(t, kind)

    def on_emit(self, t, msg):
        self.count(t, "reported_views")
        for cause in msg.snapshot.causes:
            self.count(t, "reported_cause_" + cause)

    def on_gcs_in(self, t, msg, accepted, q_len):
        self.count(t, "gcs_received")
        self.count(t, "gcs_accepted" if accepted else "gcs_stale")
        self.count(t, "gcs_received_feedback" if msg.wave == 2 else "gcs_received_pg")

    def on_fanout(self, start, end, msg):
        self.count(end, "fanout_completed")
        self.latencies["gcs_executor_wait_s"].append(start-msg.gcs_recv_t)

    def on_write(self, t, dst, msg, enqueued, start, delivered, backlog):
        self.count(t, "gcs_broadcast_recipient_writes")
        self.count(t, "gcs_broadcast_feedback" if msg.wave == 2 else "gcs_broadcast_pg")
        self.latencies["sync_delivery_s"].append(delivered-msg.emit_t)
        self.latencies["sender_queue_s"].append(start-enqueued)
        self.single_queue_peak = max(self.single_queue_peak, backlog)

    def on_recv(self, t, dst, msg):
        self.count(t, "raylet_applied")

    def on_sample(self, t, data):
        self.all_queues_peak = max(self.all_queues_peak, data["link_backlog_total"])
        self.single_queue_peak = max(self.single_queue_peak, data["link_backlog_max"])

    def rows(self, end):
        keys = sorted(set().union(*(x.keys() for x in self.bins.values())))
        return [dict(bin_start_s=i, bin_end_s=i+1,
                     **{k:self.bins[i].get(k, 0) for k in keys})
                for i in range(max(1, math.ceil(end-self.origin)+1))]


class PGSimulator(Simulator):
    """Original Push transport + serial GCS PG scheduling + resource probes.

    Reuses Reactor's original single-message sending path, filtering and costs.
    GCS PG control has its own serialized service lane. Ray's PG manager limits
    scheduling to one in-progress PG (manager.cc L300-348). Local processing is
    serialized per Raylet. Failures here mean resource shortage, not node loss.
    """

    def __init__(self, cfg: PGConfig):
        super().__init__(cfg, recorder=None)
        self.cfg = cfg
        self.rec = PGRecorder(cfg.wave1_start_s)
        self.received_versions.fill(0)  # ordinary pre-PG resource discovery is warm.
        self.resources = [NodeResources(cfg.cpus_per_node, cfg.node_resource_capacity,
                                        cfg.cpus_per_node, cfg.node_resource_capacity)
                          for _ in range(self.N)]
        self.pgs = {}
        self.owner_pgs = {}
        self.ready_tasks = {}
        self.pg_pending = deque()
        self.pg_busy = False
        self.gcs_control_free = 0.0
        self.node_free = [0.0] * self.N
        self.report_pending = set()
        self.reported_version = [0] * self.N
        self.causes = [Counter() for _ in range(self.N)]
        self.report_phase = [(self.rng.random()*cfg.report_period_s
                              if cfg.stagger_report_phase else 0.0) for _ in range(self.N)]
        self.views = np.empty((self.N, self.N), dtype=object)
        self.gcs_snapshots = [self._snapshot(i, ()) for i in range(self.N)]
        for src in range(self.N):
            self.views[:, src] = self.gcs_snapshots[src]
        self.try_pending = set()
        self.targets = [i for i in range(self.N) if not cfg.skip_head or i != cfg.driver_node]
        self.submit_complete_s = None
        self.coverage_count = self.N*(self.N-1)
        self.min_cpu_free = cfg.cpus_per_node
        self.min_token_free = BUNDLE_CAPACITY
        self.resource_checks = 0
        self.schedule_without_view = 0
        self.stale_applies = 0
        self.control_messages = 0
        self.last_business_s = cfg.wave1_start_s
        self.last_apply_s = cfg.wave1_start_s
        self.last_emit_s = cfg.wave1_start_s
        self.stop_reason = "not_started"

    @staticmethod
    def bundle_key(pg_id, index=-1):
        return f"bundle_group_{pg_id}" if index < 0 else f"bundle_group_{index}_{pg_id}"

    @staticmethod
    def cpu_key(pg_id, index=-1):
        return f"CPU_group_{pg_id}" if index < 0 else f"CPU_group_{index}_{pg_id}"

    def _snapshot(self, node, causes):
        r = self.resources[node]
        total = {"CPU":r.capacity_cpu, f"node:{node}":r.capacity_node, **r.pg_total}
        available = {"CPU":r.free_cpu, f"node:{node}":r.free_node, **r.pg_available}
        return ResourceSnapshot(node, self.local_version[node], tuple(sorted(total.items())),
                                tuple(sorted(available.items())), tuple(sorted(causes)))

    def _node_job(self, node, kind, payload, cost_us=None, delay=0.0):
        cost = self.cfg.node_proc_us if cost_us is None else cost_us
        end = max(self.now+delay, self.node_free[node]) + cost*US
        self.node_free[node] = end
        self._push(end, kind, payload)

    def _gcs_job(self, kind, payload, delay=0.0):
        end = max(self.now+delay, self.gcs_control_free) + self.cfg.gcs_control_us*US
        self.gcs_control_free = end
        self._push(end, kind, payload)

    def _rpc(self, destination, kind, payload, cost_us=None):
        self.control_messages += 1
        delay = self.cfg.control_oneway_s
        if destination == "gcs":
            self._gcs_job(kind, payload, delay)
        else:
            self._node_job(destination, kind, payload, cost_us, delay)

    def _changed(self, node, cause):
        # Version increments and report creation are deliberately separate.
        old = self.local_version[node]
        previously_covered = int(np.count_nonzero(self.received_versions[:, node] >= old))
        if self.received_versions[node, node] >= old:
            previously_covered -= 1
        self.coverage_count -= previously_covered
        self.local_version[node] += 1
        self.causes[node][cause] += 1
        self.rec.count(self.now, "resource_version_changes")
        self.rec.count(self.now, "changes_" + cause)
        if node not in self.report_pending:
            period = self.cfg.report_period_s
            phase = self.cfg.wave1_start_s + self.report_phase[node]
            k = max(0, math.floor((self.now-phase)/period) + 1)
            when = phase+k*period
            self.report_pending.add(node)
            self._push(max(when, self.now+1e-12), "report", node)

    def _report(self, node):
        self.report_pending.discard(node)
        if self.local_version[node] <= self.reported_version[node]:
            return
        snap = self._snapshot(node, self.causes[node])
        self.causes[node].clear()
        self.reported_version[node] = snap.version
        msg = PGMessage(next(self._mid), snap, self.now)
        self.rec.on_emit(self.now, msg)
        self.last_emit_s = self.now
        self._push(self.now+self.cfg.uplink_us*US, E_GCS_IN, msg)

    def _check_resources(self, node):
        r = self.resources[node]
        self.resource_checks += 1
        assert abs(r.free_cpu+r.reserved_cpu-r.capacity_cpu) < EPS
        assert abs(r.free_node+r.reserved_node-r.capacity_node) < EPS
        assert r.free_cpu >= -EPS and r.free_node >= -EPS
        self.min_cpu_free = min(self.min_cpu_free, r.free_cpu)
        for key, total in r.pg_total.items():
            assert -EPS <= r.pg_available[key] <= total+EPS
            if key.startswith("bundle_group_"):
                self.min_token_free = min(self.min_token_free, r.pg_available[key])

    def _register(self, target_index):
        owner = self.targets[target_index]
        pg = PGRecord(target_index, owner,
                      [BundleRecord(i, self.cfg.cpus_per_bundle, self.cfg.node_per_bundle)
                       for i in range(self.cfg.bundles_per_pg)], registered_s=self.now)
        self.pgs[pg.pg_id] = pg
        self.pg_pending.append(pg.pg_id)
        self.rec.event(self.now, "pg_registered", pg.pg_id, owner)
        self._schedule_pg()
        self.control_messages += 1
        self._push(self.now+self.cfg.control_oneway_s, "register_reply", target_index)

    def _schedule_pg(self):
        if self.pg_busy or not self.pg_pending:
            return
        self.pg_busy = True
        pg_id = self.pg_pending.popleft()
        pg = self.pgs[pg_id]
        pg.attempts += 1
        self._rpc(pg.owner, "prepare", pg_id,
                  self.cfg.bundle_op_us*len(pg.bundles))

    def _prepare(self, pg_id):
        pg = self.pgs[pg_id]
        r = self.resources[pg.owner]
        prepared = []
        for b in pg.bundles:
            if r.free_cpu+EPS < b.cpu or r.free_node+EPS < b.node:
                break
            r.free_cpu -= b.cpu
            r.free_node -= b.node
            r.reserved_cpu += b.cpu
            r.reserved_node += b.node
            b.state = "PREPARED"
            prepared.append(b)
            self._changed(pg.owner, "pg_prepare")
            self._check_resources(pg.owner)
        ok = len(prepared) == len(pg.bundles)
        if not ok:
            for b in prepared:
                r.free_cpu += b.cpu
                r.free_node += b.node
                r.reserved_cpu -= b.cpu
                r.reserved_node -= b.node
                b.state = "PENDING"
                self._changed(pg.owner, "pg_rollback")
            self._check_resources(pg.owner)
        self.rec.event(self.now, "pg_prepared" if ok else "pg_prepare_failed", pg_id, pg.owner)
        self._rpc("gcs", "prepare_reply", (pg_id, ok))

    def _prepare_reply(self, pg_id, ok):
        pg = self.pgs[pg_id]
        if ok:
            pg.state = "PREPARED"
            self._rpc(pg.owner, "commit", pg_id,
                      self.cfg.bundle_op_us*len(pg.bundles))
        else:
            self.pg_busy = False
            if pg.attempts < self.cfg.max_pg_attempts:
                self._push(self.now+self.cfg.retry_backoff_s, "retry_pg", pg_id)
            else:
                pg.state = "FAILED"
                pg.failed_s = self.now
                if pg_id in self.ready_tasks:
                    self.ready_tasks[pg_id].status = "PG_FAILED"
                self.rec.event(self.now, "pg_failed", pg_id, pg.owner)
            self._schedule_pg()

    def _commit(self, pg_id):
        pg = self.pgs[pg_id]
        r = self.resources[pg.owner]
        for b in pg.bundles:
            assert b.state == "PREPARED"
            b.state = "COMMITTED"
            for name, amount in (("CPU", b.cpu), (f"node:{pg.owner}", b.node),
                                 ("bundle", BUNDLE_CAPACITY)):
                for index in (b.index, -1):
                    key = (self.bundle_key(pg_id,index) if name == "bundle"
                           else f"{name}_group_{pg_id}" if index == -1
                           else f"{name}_group_{index}_{pg_id}")
                    r.pg_total[key] = r.pg_total.get(key, 0.0)+amount
                    r.pg_available[key] = r.pg_available.get(key, 0.0)+amount
                    self._changed(pg.owner, "pg_commit")
        self._check_resources(pg.owner)
        self.rec.event(self.now, "pg_committed", pg_id, pg.owner)
        self._rpc("gcs", "commit_reply", pg_id)
        # Commit triggers local lease rechecking in upstream NodeManager.
        if pg.owner == self.cfg.driver_node or self.cfg.scheduler_view == "oracle":
            self._request_try(pg_id)

    def _commit_reply(self, pg_id):
        pg = self.pgs[pg_id]
        pg.state, pg.created_s = "CREATED", self.now
        self.rec.event(self.now, "pg_created", pg_id, pg.owner)
        self.pg_busy = False
        self._schedule_pg()

    def _ready_submit(self):
        for pg_id, pg in self.pgs.items():
            self.ready_tasks[pg_id] = ReadyTask(pg_id, submitted_s=self.now,
                                               status="PG_FAILED" if pg.state == "FAILED" else "PENDING")
            self.rec.event(self.now, "ready_submitted", pg_id, self.cfg.driver_node)
            self._request_try(pg_id)

    def _request_try(self, pg_id):
        task = self.ready_tasks.get(pg_id)
        if task is None or task.status != "PENDING" or pg_id in self.try_pending:
            return
        self.try_pending.add(pg_id)
        self._node_job(self.cfg.driver_node, "try_ready", pg_id, self.cfg.schedule_us)

    def _try_ready(self, pg_id):
        self.try_pending.discard(pg_id)
        task = self.ready_tasks[pg_id]
        if task.status != "PENDING":
            return
        pg = self.pgs[pg_id]
        task.attempts += 1
        driver = self.cfg.driver_node
        if pg.owner == driver or self.cfg.scheduler_view == "oracle":
            snap = self._snapshot(pg.owner, ())
        else:
            snap = self.views[driver, pg.owner]
        amount = snap.available_amount(self.bundle_key(pg_id))
        if amount+EPS < READY_AMOUNT:
            self.rec.count(self.now, "ready_view_blocked")
            return
        task.decision_version = snap.version
        task.decision_bundle_available = amount
        task.status = "LEASE_REQUESTED"
        self.rec.event(self.now, "ready_schedulable", pg_id, driver, view_version=snap.version)
        self._rpc(pg.owner, "lease_grant", pg_id)

    def _lease_grant(self, pg_id):
        pg, task = self.pgs[pg_id], self.ready_tasks[pg_id]
        r = self.resources[pg.owner]
        key = self.bundle_key(pg_id)
        if r.pg_available.get(key, 0.0)+EPS < READY_AMOUNT:
            # Future workloads may race; recheck both snapshot and true resources.
            task.status = "PENDING"
            self.rec.event(self.now, "lease_rejected", pg_id, pg.owner)
            self._push(self.now+self.cfg.retry_backoff_s, "retry_task", pg_id)
            return
        assert all(b.state == "COMMITTED" for b in pg.bundles)
        r.pg_available[key] -= READY_AMOUNT
        task.status, task.granted_s = "RUNNING", self.now
        self._changed(pg.owner, "lease_allocate")
        self._check_resources(pg.owner)
        self.rec.event(self.now, "lease_granted", pg_id, pg.owner)
        self._push(self.now+self.cfg.ready_runtime_s, "task_execution_done", pg_id)

    def _execution_done(self, pg_id):
        pg = self.pgs[pg_id]
        self.rec.event(self.now, "ready_executed", pg_id, pg.owner)
        self._rpc(self.cfg.driver_node, "ready_result", pg_id)
        # Explicit one-task-per-lease abstraction, not a claim about Ray's lease pool.
        self._node_job(pg.owner, "lease_release", pg_id, delay=self.cfg.lease_hold_s)

    def _release(self, pg_id):
        pg, task = self.pgs[pg_id], self.ready_tasks[pg_id]
        self.resources[pg.owner].pg_available[self.bundle_key(pg_id)] += READY_AMOUNT
        task.released_s = self.now
        self._changed(pg.owner, "lease_release")
        self._check_resources(pg.owner)
        self.rec.event(self.now, "lease_released", pg_id, pg.owner)

    def _on_gcs_in(self, msg):
        super()._on_gcs_in(msg)

    def _on_exec_done(self, task):
        if task[0] == E_FANOUT:
            msg = task[1]
            if msg.version > self.gcs_snapshots[msg.src].version:
                self.gcs_snapshots[msg.src] = msg.snapshot
        super()._on_exec_done(task)

    def _schedule_receive(self, dst, messages):
        for msg in messages:
            self._on_recv(dst, msg)

    def _on_recv(self, dst, msg):
        self.rec.count(self.now, "raylet_arrived")
        self._node_job(dst, "apply_view", (dst,msg))

    def _apply(self, dst, msg):
        old = self.received_versions[dst,msg.src]
        if msg.version <= old:
            self.stale_applies += 1
            return
        self.views[dst,msg.src] = msg.snapshot
        self.received_versions[dst,msg.src] = msg.version
        if msg.version == self.local_version[msg.src] and old < msg.version:
            self.coverage_count += 1
        self.latest_version_recv[dst,msg.src] = self.now
        if self.first_recv[dst,msg.src] < 0:
            self.first_recv[dst,msg.src] = self.now
        self.last_recv[dst,msg.src] = self.now
        self.last_apply_s = self.now
        self.rec.on_recv(self.now,dst,msg)
        if dst == self.cfg.driver_node:
            # Fixed-node workload: at most one PG per target, but use explicit IDs.
            for pg_id in self.owner_pgs.get(msg.src, ()):
                self._request_try(pg_id)

    def _dispatch_business(self, kind, payload):
        if kind == "register":
            self._register(payload)
        elif kind == "register_reply":
            if payload+1 < len(self.targets):
                self._rpc("gcs", "register", payload+1)
            else:
                self.submit_complete_s = self.now
                self.owner_pgs = {pg.owner:[pg_id] for pg_id,pg in self.pgs.items()}
                if self.cfg.submit_ready:
                    self._push(self.now+self.cfg.wait_before_ready_s, "ready_submit", None)
        elif kind == "prepare": self._prepare(payload)
        elif kind == "prepare_reply": self._prepare_reply(*payload)
        elif kind == "commit": self._commit(payload)
        elif kind == "commit_reply": self._commit_reply(payload)
        elif kind == "retry_pg":
            self.pg_pending.append(payload)
            self._schedule_pg()
        elif kind == "ready_submit": self._ready_submit()
        elif kind == "try_ready": self._try_ready(payload)
        elif kind == "retry_task": self._request_try(payload)
        elif kind == "lease_grant": self._lease_grant(payload)
        elif kind == "task_execution_done": self._execution_done(payload)
        elif kind == "ready_result":
            task = self.ready_tasks[payload]
            task.status, task.done_s = "DONE", self.now
            self.rec.event(self.now, "ready_done", payload, self.cfg.driver_node)
        elif kind == "lease_release": self._release(payload)
        elif kind == "report": self._report(payload)
        elif kind == "apply_view": self._apply(*payload)
        else: raise ValueError(kind)

    def run(self):
        self.owner_pgs = {}
        self.now = self.cfg.wave1_start_s
        self._rpc("gcs", "register", 0)
        self._push_sample(self.now+1.0)
        while self.heap:
            t, seq, kind, payload = heapq.heappop(self.heap)
            if t > self.cfg.max_sim_s:
                self.now = self.cfg.max_sim_s
                self.stop_reason = "max_sim_s"
                break
            self.now = t
            if kind == CALLBACK:
                fn, args = payload
                fn(*args)
            elif kind == E_GCS_IN: self._on_gcs_in(payload)
            elif kind == E_EXEC_DONE: self._on_exec_done(payload)
            elif kind == E_RECV: self._on_recv(*payload)
            elif kind == E_SAMPLE:
                self._on_sample()
                if self.heap: self._push_sample(self.now+1.0)
                continue
            else:
                self._dispatch_business(kind,payload)
                if kind not in ("report", "apply_view"):
                    self.last_business_s = self.now
        else:
            self.stop_reason = "quiescent"
        return self.summary()

    @staticmethod
    def _distribution(values):
        values = list(values)
        if not values: return dict(count=0, mean=None, p50=None, p95=None, max=None)
        return dict(count=len(values),mean=float(np.mean(values)),p50=float(np.median(values)),
                    p95=float(np.percentile(values,95)),max=float(max(values)))

    def summary(self):
        mask = ~np.eye(self.N,dtype=bool)
        final = np.asarray(self.local_version)
        covered = (self.received_versions >= final[None,:]) & mask
        coverage = int(np.count_nonzero(covered))
        assert coverage == self.coverage_count, (coverage,self.coverage_count)
        totals = Counter()
        for row in self.rec.bins.values(): totals.update(row)
        writes = sum(r.writes for r in self.reactors)
        quiescent = self.stop_reason == "quiescent"
        converged = coverage == self.N*(self.N-1) and not self.report_pending and quiescent
        created = [p.created_s for p in self.pgs.values() if p.created_s is not None]
        done = [t.done_s for t in self.ready_tasks.values() if t.done_s is not None]
        released = [t.released_s for t in self.ready_tasks.values() if t.released_s is not None]
        origin = self.cfg.wave1_start_s
        elapsed = lambda xs: max(xs)-origin if xs else None
        decisions = [t for t in self.ready_tasks.values() if t.decision_version is not None]
        return dict(config=asdict(self.cfg), model="PG on Original Push", counts=dict(totals),
                    pg_target=len(self.targets), pg_created=len(created),
                    pg_failed=sum(p.state=="FAILED" for p in self.pgs.values()),
                    ready_submitted=len(self.ready_tasks), ready_done=len(done),
                    ready_waiting=sum(t.status not in ("DONE","PG_FAILED") for t in self.ready_tasks.values()),
                    leases_released=len(released), control_messages=self.control_messages,
                    pg_all_created_s=elapsed(created) if len(created)==len(self.targets) else None,
                    ready_all_done_s=elapsed(done) if len(done)==len(self.targets) else None,
                    ready_wait_after_submission_s=(max(done)-min(t.submitted_s for t in self.ready_tasks.values()) if done else None),
                    final_sync_convergence_s=(self.last_apply_s-origin if converged else None),
                    last_resource_report_s=self.last_emit_s-origin,
                    simulation_end_s=self.now-origin, stop_reason=self.stop_reason,
                    converged=converged,final_version_coverage=coverage/(self.N*(self.N-1)),
                    uncovered_final_pairs=self.N*(self.N-1)-coverage,
                    total_gcs_recipient_writes=writes,total_gcs_received=totals['gcs_received'],
                    amplification=writes/max(1,totals['gcs_received']),
                    version_changes=sum(self.local_version),resource_checks=self.resource_checks,
                    min_cpu_available=self.min_cpu_free,min_bundle_available=self.min_token_free,
                    resource_conservation="PASS",schedule_decisions=len(decisions),
                    view_constraint_check=all(t.decision_bundle_available+EPS>=READY_AMOUNT for t in decisions),
                    peak_single_link_buffer=self.rec.single_queue_peak,
                    peak_total_link_buffer_sampled=self.rec.all_queues_peak,
                    inplace_merges=self.inplace_merges,
                    metrics={k:self._distribution(v) for k,v in self.rec.latencies.items()},
                    ready_latency_s=self._distribution(t.done_s-t.submitted_s for t in self.ready_tasks.values() if t.done_s is not None),
                    pg_creation_latency_s=self._distribution(p.created_s-p.registered_s for p in self.pgs.values() if p.created_s is not None),
                    seed=self.cfg.seed)
