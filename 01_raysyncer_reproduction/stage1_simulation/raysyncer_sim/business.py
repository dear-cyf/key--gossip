"""Fixed-target task execution; resource reports are effects, never inputs."""
from collections import deque
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Resources:
    cpu_total: float
    cpu_available: float
    memory_total_mib: float
    memory_available_mib: float


@dataclass
class Task:
    task_id: int
    submit_s: float
    node: int
    task_class: str
    cpu: float
    memory_mib: float
    work_cpu_s: float
    duration_s: float
    status: str = 'not_submitted'
    start_s: float | None = None
    finish_s: float | None = None
    cpu_cost_s: float = 0.
    memory_cost_mib_s: float = 0.


def normalize_task_spec(spec, tid):
    """Accept historical five-column inputs and the explicit eight-column table."""
    if len(spec) == 5:
        t, node, cpu, mem, duration = spec
        return (tid, t, node, 'legacy', cpu, mem, cpu * duration, duration)
    if len(spec) != 8:
        raise ValueError('task must have five legacy columns or eight task-table columns')
    task_id, t, node, kind, cpu, mem, work, duration = spec
    if (not isinstance(task_id, int) or task_id != tid or not isinstance(kind, str) or not kind
            or not math.isfinite(work) or work < 0
            or not math.isclose(work, cpu * duration, rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError('invalid task identity or work_cpu_s != cpu * duration_s')
    return tuple(spec)


class Business:
    def __init__(self, sim):
        self.sim = sim
        c = sim.cfg
        self.resources = [Resources(c.cpus_per_node, c.cpus_per_node,
                                    c.memory_per_node_mib, c.memory_per_node_mib)
                          for _ in range(sim.N)]
        sim.resource_state = list(self.resources)
        sim.resource_history = {(node, 0): state for node, state in enumerate(self.resources)}
        # Warm ordinary resource discovery is the initial condition, not an update.
        sim.replica_resources = np.empty((sim.N, sim.N), dtype=object)
        for node, state in enumerate(self.resources):
            sim.replica_resources[:, node] = state
        self.queues = [deque() for _ in range(sim.N)]
        self.running = [{} for _ in range(sim.N)]
        self.tasks = []
        self.events = []
        self.changes = []
        self.observations = []
        self.done = 0
        self.last_finish_s = c.wave1_start_s
        specs = c.business_tasks
        if specs is None:
            specs = tuple((c.wave1_start_s + node*c.wave1_emit_spread_s/sim.N
                           + k*c.task_interval_s, node, c.task_cpu,
                           c.task_memory_mib, c.task_duration_s)
                          for node in range(sim.N) for k in range(c.tasks_per_node))
        for tid, spec in enumerate(specs):
            normalized = normalize_task_spec(spec, tid)
            _, t, node, kind, cpu, mem, work, duration = normalized
            if (not isinstance(node, int) or not 0 <= node < sim.N
                    or any(not math.isfinite(v) or v < 0 for v in (t,cpu,mem,duration))):
                raise ValueError('invalid task specification')
            self.tasks.append(Task(*normalized))
        self.start_s = min((t.submit_s for t in self.tasks), default=c.wave1_start_s)
        self.submitted_count = len(self.tasks)
        sim.workload_last_s = math.inf if self.tasks else c.wave1_start_s
        # Full initial versions are known; first-receive timings remain separate.
        if hasattr(sim, 'received_versions'):
            sim.received_versions.fill(0)
            sim.latest_version_recv.fill(self.start_s)
        else:
            sim.view_version.fill(0)
            sim.have_final_counts.fill(sim.N-1)
            sim.last_accept_t.fill(self.start_s)
        if hasattr(sim.rec, 'on_initial_resources'):
            sim.rec.on_initial_resources(self.start_s, self.resources)

    @property
    def complete(self):
        return self.done == len(self.tasks)

    def schedule(self):
        for task in self.tasks:
            self.sim.net.later(task.submit_s, self.submit, task.task_id)

    def log(self, kind, task):
        self.events.append(dict(time_s=self.sim.now, event=kind, task_id=task.task_id,
                                node=task.node, status=task.status))

    def submit(self, tid):
        task = self.tasks[tid]
        self.log('submit',task)
        state = self.resources[task.node]
        if task.cpu > state.cpu_total or task.memory_mib > state.memory_total_mib:
            task.status = 'rejected_capacity'
            self.log('reject',task)
            self._finished()
            return
        task.status = 'queued'
        self.queues[task.node].append(task)
        self._dispatch(task.node)

    def _dispatch(self, node):
        queue = self.queues[node]
        while queue:
            task = queue[0]
            state = self.resources[node]
            if task.cpu > state.cpu_available + 1e-10 or task.memory_mib > state.memory_available_mib + 1e-10:
                self.log('wait_resources',task)
                return
            queue.popleft()
            task.status = 'running'
            task.start_s = self.sim.now
            self.log('start',task)
            self.running[node][task.task_id] = task
            self._mutate(task, 'allocate')
            # Runtime starts only after real admission, not at submit time.
            self.sim.net.later(self.sim.now + task.duration_s, self.finish, task.task_id)

    def finish(self, tid):
        task = self.tasks[tid]
        if task.status != 'running':
            raise RuntimeError('completion without a running task')
        task.status = 'completed'
        task.finish_s = self.sim.now  #记录完成
        self.log('finish',task)
        del self.running[task.node][task.task_id]
        self._mutate(task, 'release')   #释放资源
        self._dispatch(task.node)      #调度下一任务
        self._finished()

    def _mutate(self, task, cause):
        before = self.resources[task.node]
        reservations = self.running[task.node].values()
        cpu = round(before.cpu_total - math.fsum(t.cpu for t in reservations), 12)
        memory = round(before.memory_total_mib - math.fsum(t.memory_mib for t in reservations), 12)
        after = Resources(before.cpu_total, cpu, before.memory_total_mib, memory)
        if not (0 <= after.cpu_available <= after.cpu_total
                and 0 <= after.memory_available_mib <= after.memory_total_mib):
            raise RuntimeError('resource conservation violated')
        # Write actual allocation state, then run the observer; no version/count input.
        self.resources[task.node] = after
        changed = self.sim.resource_change(task.node, after)
        record = dict(time_s=self.sim.now,node=task.node,task_id=task.task_id,
                cause=cause,version=self.sim.local_version[task.node],changed=changed,
                cpu_before=before.cpu_available,cpu_after=after.cpu_available,
                memory_before_mib=before.memory_available_mib,memory_after_mib=after.memory_available_mib)
        self.observations.append(record)
        if changed:
            self.changes.append(record)

    def _finished(self):
        self.done += 1
        self.last_finish_s = self.sim.now
        if self.complete:
            self.sim.workload_last_s = self.sim.now

    def summary(self):
        # Reservation-time costs include only elapsed execution, never queueing.
        # At a cutoff, charge running tasks up to the observation boundary.
        for task in self.tasks:
            elapsed = (max(0., min(task.duration_s,
                       (task.finish_s if task.finish_s is not None else self.sim.now) - task.start_s))
                       if task.start_s is not None else 0.)
            task.cpu_cost_s = task.cpu * elapsed
            task.memory_cost_mib_s = task.memory_mib * elapsed
        return dict(tasks=len(self.tasks),completed=sum(t.status=='completed' for t in self.tasks),
                    cpu_cost_s=math.fsum(t.cpu_cost_s for t in self.tasks),
                    memory_cost_mib_s=math.fsum(t.memory_cost_mib_s for t in self.tasks),
                    requested_work_cpu_s=math.fsum(t.work_cpu_s for t in self.tasks),
                    rejected=sum(t.status=='rejected_capacity' for t in self.tasks),
                    queued=sum(t.status=='queued' for t in self.tasks),
                    running=sum(t.status=='running' for t in self.tasks),
                    not_submitted=sum(t.status=='not_submitted' for t in self.tasks),
                    resource_changes=len(self.changes),business_complete=self.complete,
                    last_business_event_s=self.last_finish_s,
                    all_resources_released=all(s.cpu_available==s.cpu_total
                        and s.memory_available_mib==s.memory_total_mib for s in self.resources))
