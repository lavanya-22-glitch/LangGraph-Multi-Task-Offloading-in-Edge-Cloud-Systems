# agentic_offloading/core/workflow.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Task:
    """
    Represents task node v_i (as in the paper):
      • task_id i ∈ [0..N+1]
      • v_i  : CPU cycles
      • deps : {j: d_{i,j}} where d_{i,j} is data size in bytes
    """
    task_id: int
    v_i: float
    deps: Dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"task_id": self.task_id, "v": self.v_i, "deps": dict(self.deps)}


class Workflow:
    """
    Workflow DAG  w = {V, D}
      • V = {v_i : 0 ≤ i ≤ N+1}
      • D = {d_{i,j} : 0 ≤ i, j ≤ N+1}

    Constructed directly from the experiment input:

        workflow: {
          "tasks": {1: {"v": int}, 2:{...}, ..., N:{...}},   # CPU cycles
          "edges": {(i,j): int, ...},                        # bytes (may omit entry/exit)
          "N": int
        }

    Automatically adds:
      • Entry node v_0 and Exit node v_{N+1} (unit tasks, non-offloadable)
      • Missing edges (0 → source) and (sink → N+1) with 0-byte weights
    """

    def __init__(self) -> None:
        """Initialize an empty Workflow (internal use only)."""
        self._tasks: List[Task] = []

    # -------------------------- construction API --------------------------

    @classmethod
    def from_experiment_dict(cls, workflow_dict: Dict) -> "Workflow":
        """
        Build workflow object directly from the input structure defined in the paper.
        """
        obj = cls()

        tasks_map: Dict[int, Dict[str, float]] = workflow_dict["tasks"]
        edges_map: Dict[Tuple[int, int], float] = workflow_dict.get("edges", {})
        N: int = int(workflow_dict["N"])

        if N <= 0:
            raise ValueError("N must be >= 1.")

        # --- Create entry/exit ---
        entry_id, exit_id = 0, N + 1
        entry = Task(task_id=entry_id, v_i=1.0, deps={})
        exit_task = Task(task_id=exit_id, v_i=1.0, deps={})

        # --- Create real tasks ---
        id_to_task: Dict[int, Task] = {i: Task(i, float(tasks_map[i]["v"])) for i in range(1, N + 1)}

        # --- Assign edges ---
        provided_entry, provided_exit = set(), set()
        for (i, j), dij in edges_map.items():
            i, j = int(i), int(j)
            d = float(dij)

            if i == entry_id and 1 <= j <= N:
                entry.deps[j] = d
                provided_entry.add(j)
            elif 1 <= i <= N and j == exit_id:
                id_to_task[i].deps[exit_id] = d
                provided_exit.add(i)
            elif 1 <= i <= N and 1 <= j <= N:
                id_to_task[i].deps[j] = d
            else:
                # Ignore invalid edges (outside 0..N+1)
                continue

        # --- Compute parents/children to find sources/sinks ---
        parents = {i: [] for i in range(1, N + 1)}
        children = {i: [] for i in range(1, N + 1)}
        for i in range(1, N + 1):
            for j in id_to_task[i].deps:
                if 1 <= j <= N:
                    parents[j].append(i)
                    children[i].append(j)

        # --- Add missing entry edges (0→sources) ---
        for i in range(1, N + 1):
            if len(parents[i]) == 0 and i not in provided_entry:
                entry.deps[i] = 0.0

        # --- Add missing exit edges (sinks→N+1) ---
        for i in range(1, N + 1):
            if len(children[i]) == 0 and i not in provided_exit:
                id_to_task[i].deps[exit_id] = 0.0

        # --- Finalize list of tasks ---
        obj._tasks = [entry] + [id_to_task[i] for i in range(1, N + 1)] + [exit_task]

        # --- Validate DAG ---
        obj._validate_acyclic()

        return obj

    # ----------------------------- properties -----------------------------

    @property
    def N(self) -> int:
        """Number of real tasks (excluding entry/exit)."""
        return len(self._tasks) - 2

    def vertices(self) -> List[int]:
        """All vertex ids 0..N+1."""
        return [t.task_id for t in self._tasks]

    def V(self) -> Dict[int, float]:
        """Map i -> v_i (CPU cycles)."""
        return {t.task_id: t.v_i for t in self._tasks}

    def D(self) -> Dict[Tuple[int, int], float]:
        """Map (i,j) -> d_{i,j} (bytes)."""
        edges: Dict[Tuple[int, int], float] = {}
        for t in self._tasks:
            for j, dij in t.deps.items():
                edges[(t.task_id, j)] = float(dij)
        return edges

    def Ji(self) -> Dict[int, List[int]]:
        """Parents of each node i."""
        ji: Dict[int, List[int]] = {t.task_id: [] for t in self._tasks}
        for t in self._tasks:
            for j in t.deps:
                ji[j].append(t.task_id)
        return ji

    def Ki(self) -> Dict[int, List[int]]:
        """Children of each node i."""
        ki: Dict[int, List[int]] = {t.task_id: [] for t in self._tasks}
        for t in self._tasks:
            for j in t.deps:
                ki[t.task_id].append(j)
        return ki

    def to_experiment_dict(self) -> Dict:
        """
        Export back into the experiment input shape
        (real tasks only in 'tasks'; edges include entry/exit).
        """
        tasks_map = {i: {"v": self.V()[i]} for i in range(1, self.N + 1)}
        edges_map = {}
        for (i, j), d in self.D().items():
            edges_map[(i, j)] = d
        return {"tasks": tasks_map, "edges": edges_map, "N": self.N}

    # ----------------------------- validation -----------------------------

    def _validate_acyclic(self) -> None:
        """Check DAG structure via DFS."""
        adj = self.Ki()
        color = {i: 0 for i in self.vertices()}  # 0=unvisited,1=visiting,2=done

        def dfs(u: int) -> bool:
            color[u] = 1
            for v in adj.get(u, []):
                if color[v] == 1:
                    return False
                if color[v] == 0 and not dfs(v):
                    return False
            color[u] = 2
            return True

        for u in self.vertices():
            if color[u] == 0 and not dfs(u):
                raise ValueError("Workflow contains a cycle.")
