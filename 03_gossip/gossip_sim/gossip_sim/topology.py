"""Deterministic k-neighbour overlay + per-message fan-out selection.

Design (``gossip_design/01_design.md`` §3.1/§3.4):

* every node keeps a **ring successor** (guarantees strong connectivity even in
  the worst case) plus ``k-1`` extra out-neighbours ranked by a *symmetric*
  rendezvous hash, so neighbours survive membership churn and the experiment is
  reproducible without any global coordination;
* a relay of message ``m`` goes to ``f`` of the node's out-neighbours, ranked by
  a per-message hash so consecutive updates do not always take the same path.

Only the *protocol* lives here; the cost of a relay is paid by the sender's own
executor in ``core.py``, exactly like the C++ ``RaySyncerBidiReactorBase``.
"""

from __future__ import annotations

import random
from typing import Iterable, List, Sequence

_MASK64 = (1 << 64) - 1


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (x ^ (x >> 31)) & _MASK64


def pair_hash(a: int, b: int, seed: int) -> int:
    """Symmetric weight of the unordered pair (a, b)."""
    lo, hi = (a, b) if a <= b else (b, a)
    return _splitmix64((lo << 32) ^ hi ^ (seed * 0x100000001B3))


def msg_target_hash(mid: int, node: int, seed: int) -> int:
    return _splitmix64((mid << 20) ^ node ^ (seed * 0xD1B54A32D192ED03))


class GossipTopology:
    """Out-neighbour sets of a directed, strongly-connected overlay."""

    def __init__(self, num_nodes: int, degree_k: int, topology_seed: int,
                 extra_neighbours: int = 0, neighbor_mode: str = "rendezvous"):
        if num_nodes < 3:
            raise ValueError("num_nodes must be >= 3")
        self.n = int(num_nodes)
        #: out-degree actually realized (ring + hashed extras, capped by n-1)
        self.k = min(degree_k + extra_neighbours, self.n - 1)
        if self.k < 1:
            raise ValueError("degree_k must be >= 1")
        self.seed = topology_seed
        self.neighbor_mode = neighbor_mode
        if neighbor_mode == "random":
            # A dedicated RNG: the *workload* RNG of the simulator must not be
            # touched, otherwise the star/batching arms would no longer be
            # bit-identical baselines.
            rng = random.Random(f"{topology_seed}:neighbours")
            self.out = []
            for i in range(self.n):
                succ = (i + 1) % self.n
                pool = [j for j in range(self.n) if j != i and j != succ]
                rng.shuffle(pool)
                self.out.append([succ] + pool[: self.k - 1])
        elif neighbor_mode == "rendezvous":
            self.out = [self._build(i) for i in range(self.n)]
        else:
            raise ValueError(f"unknown neighbor_mode: {neighbor_mode}")
        # bookkeeping for the report
        self.ring_edges = self.n

    def _build(self, i: int) -> List[int]:
        n = self.n
        succ = (i + 1) % n
        picked = [succ]
        cands = [j for j in range(n) if j != i and j != succ]
        cands.sort(key=lambda j: pair_hash(i, j, self.seed), reverse=True)
        for j in cands:
            if len(picked) >= self.k:
                break
            picked.append(j)
        return picked

    # ------------------------------------------------------------------ #
    def neighbours(self, node: int) -> Sequence[int]:
        return self.out[node]

    def select_fanout(self, node: int, mid: int, src: int,
                      exclude: Iterable[int], fanout_f: int,
                      mode: str = "hash", rng: random.Random | None = None) -> List[int]:
        """Pick ``fanout_f`` targets for one relay.

        Mirrors ``RaySyncerBidiReactorBase::PushToSendingQueue``: never send a
        message back to the node it originates from (``src``); the caller may
        additionally exclude the neighbour the message arrived from.
        """
        blocked = {src}
        blocked.update(exclude)
        cands = [j for j in self.out[node] if j not in blocked]
        if not cands:
            return []
        if len(cands) > fanout_f:
            if mode == "random" and rng is not None:
                cands = rng.sample(cands, fanout_f)
            else:
                cands.sort(key=lambda j: msg_target_hash(mid, j, self.seed),
                           reverse=True)
                cands = cands[:fanout_f]
        return cands

    # ------------------------------------------------------------------ #
    def bfs_diameter(self) -> int:
        """Diameter of the undirected view of the overlay (report only)."""
        n = self.n
        adj = [set() for _ in range(n)]
        for i, outs in enumerate(self.out):
            for j in outs:
                adj[i].add(j)
                adj[j].add(i)
        best = 0
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            frontier = [s]
            while frontier:
                nxt = []
                for u in frontier:
                    for v in adj[u]:
                        if dist[v] < 0:
                            dist[v] = dist[u] + 1
                            nxt.append(v)
                frontier = nxt
            if any(d < 0 for d in dist):
                return -1  # disconnected
            best = max(best, max(dist))
        return best

    def summary(self) -> dict:
        degs = [len(o) for o in self.out]
        return {
            "n": self.n,
            "k": self.k,
            "out_degree_min": min(degs),
            "out_degree_max": max(degs),
            "directed_edges": int(sum(degs)),
            "undirected_diameter": self.bfs_diameter(),
            "topology_seed": self.seed,
            "neighbor_mode": self.neighbor_mode,
            "mutual_edge_ratio": self.mutual_edge_ratio(),
        }

    def mutual_edge_ratio(self) -> float:
        """Share of directed edges that are reciprocated (report only)."""
        outs = [set(o) for o in self.out]
        directed = sum(len(o) for o in outs)
        if directed == 0:
            return 0.0
        mutual = sum(1 for i in range(self.n) for j in outs[i] if i in outs[j])
        return mutual / directed
