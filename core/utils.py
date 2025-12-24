# agentic_offloading/core/utils.py
from typing import Dict, List, Optional
from collections import deque


def topological_sort(adj: Dict[int, List[int]]) -> Optional[List[int]]:
    """
    Return a topological ordering of the DAG described by adj (parent -> children).
    If graph contains a cycle, returns None.
    """
    # compute in-degree
    in_degree = {u: 0 for u in adj.keys()}
    for u, children in adj.items():
        for v in children:
            in_degree.setdefault(v, 0)
            in_degree[v] += 1

    q = deque([u for u, deg in in_degree.items() if deg == 0])
    order: List[int] = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)

    if len(order) != len(in_degree):
        # cycle exists
        return None
    return order
