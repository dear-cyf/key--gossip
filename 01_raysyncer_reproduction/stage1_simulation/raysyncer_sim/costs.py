"""Shared finite NIC queues and explicit local-resource workload.

Rates are application bytes/s (full duplex, one shared NIC per participant).
The default rates and costs are assumptions, not hardware measurements.
"""
from collections import defaultdict
import math

CALLBACK = "callback"


class Network:
    def __init__(self, sim):
        self.sim = sim
        self.tx_free = defaultdict(float)
        self.rx_free = defaultdict(float)
        self.stats = defaultdict(float)
        self.kinds = defaultdict(lambda: [0, 0, 0])
        self.sent = defaultdict(lambda: [0, 0, 0])
        self.received = defaultdict(lambda: [0, 0, 0])
        self.bins = defaultdict(lambda: [0, 0, 0])
        self.trace = []
        self.active = 0

    def later(self, t, fn, *args):
        self.sim._push(t, CALLBACK, (fn, args))

    def rate(self, node):
        c = self.sim.cfg
        return (c.gcs_bandwidth_mbps if node == self.sim.N else c.node_bandwidth_mbps) * 1e6 / 8

    def send(self, src, dst, size, latency, fn, args=(), messages=1, kind="data",
             *, on_tx_done=None):
        t = self.sim.now
        start = max(t, self.tx_free[src])
        end = start + size / self.rate(src)
        self.tx_free[src] = end
        wait = start - t
        self.stats["tx_wait_s"] += wait
        self.stats["tx_wait_max_s"] = max(self.stats["tx_wait_max_s"], wait)
        self.stats["tx_serialization_s"] += end - start
        self.stats["frames_sent"] += 1
        self.stats["bytes_sent"] += size
        self.active += 1
        self.stats["outstanding_peak"] = max(self.stats["outstanding_peak"], self.active)
        for target in (self.sent[src], self.kinds[kind], self.bins[int(math.floor(start / .1))]):
            for i, v in enumerate((messages, 1, size)):
                target[i] += v
        trace = None
        if self.sim.N <= 5 or self.sim.cfg.trace_network:
            trace = dict(src=src, dst=dst, kind=kind, messages=messages, bytes=size,
                         enqueue_s=t, tx_start_s=start, tx_end_s=end)
            self.trace.append(trace)
        if on_tx_done is not None:
            self.later(end, on_tx_done)
        self.later(end + latency, self._arrive, dst, size, fn, args, messages, trace)

    def _arrive(self, dst, size, fn, args, messages, trace):
        t = self.sim.now
        start = max(t, self.rx_free[dst])
        end = start + size / self.rate(dst)
        self.rx_free[dst] = end
        if trace is not None:
            trace.update(arrival_s=t, rx_start_s=start, rx_end_s=end)
        self.stats["rx_wait_s"] += start - t
        self.stats["rx_wait_max_s"] = max(self.stats["rx_wait_max_s"], start - t)
        self.stats["rx_serialization_s"] += end - start
        self.later(end, self._finish, dst, size, fn, args, messages)

    def _finish(self, dst, size, fn, args, messages):
        self.active -= 1
        self.stats["frames_received"] += 1
        self.stats["bytes_received"] += size
        for i, v in enumerate((messages, 1, size)):
            self.received[dst][i] += v
        fn(*args)


class ResourceWorkload:
    """Only an actual change of a node's own value increments its version."""
    def resource_change(self, node, value):
        if self.cfg.workload == "business":
            if not hasattr(self, "business"):
                raise RuntimeError("business resource changes require a resource manager")
            if value != self.business.resources[node]:
                raise RuntimeError("detector must read the actual local resource manager")
        if self.resource_state[node] == value:
            return False
        self.resource_state[node] = value
        self.emit(node, wave=1 if self.local_version[node] == 0 else 2,
                  rnd=self.local_version[node])
        return True

    def capture_resources(self, msg):
        msg.resources = self.resource_state[msg.src]
        if not hasattr(self, "resource_history"):
            self.resource_history = {}
        self.resource_history[msg.src, msg.version] = msg.resources
        if hasattr(self, "replica_resources"):
            self.replica_resources[msg.src, msg.src] = msg.resources

    def apply_resources(self, dst, msg):
        if hasattr(self, "replica_resources") and msg.resources is not None:
            self.replica_resources[dst, msg.src] = msg.resources

    def start_wave1(self):
        c = self.cfg
        if c.workload == "business":
            from .business import Business
            self.business = Business(self)
            self.business.schedule()
            return
        changes = c.resource_changes
        if changes is None:
            changes = tuple((c.wave1_start_s + node * c.wave1_emit_spread_s / self.N,
                             node, 1) for node in range(self.N))
        self.workload_last_s = max((event[0] for event in changes), default=0.0)
        for t, node, value in changes:
            if t < 0 or not 0 <= node < self.N:
                raise ValueError("invalid local resource event")
            self.net.later(t, self.resource_change, node, value)

    def _check_milestones(self, node):
        # Receiving/forwarding a remote entry never creates a local version.
        pass
