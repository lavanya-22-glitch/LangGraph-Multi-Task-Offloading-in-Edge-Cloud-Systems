# dag_generator.py
import random
import math
import json
from collections import defaultdict
from typing import Dict, Tuple, List

# -------------------------
# Helper / Generation code
# -------------------------

def compute_height_width(n: int, alpha: float):
    # height = ceil(sqrt(n / alpha)), width = ceil(sqrt(n * alpha))
    height = math.ceil(math.sqrt(n / alpha))
    width = math.ceil(math.sqrt(n * alpha))
    return height, width

def distribute_nodes_across_levels(n: int, height: int, width: int, rng=random):
    # ensure at least one node per level, then distribute remaining
    levels = [[] for _ in range(height)]
    node_id = 1
    # put one node per level first
    for lvl in range(height):
        levels[lvl].append(node_id)
        node_id += 1
    remaining = n - height
    while remaining > 0:
        lvl = rng.randrange(0, height)
        if len(levels[lvl]) < width:
            levels[lvl].append(node_id)
            node_id += 1
            remaining -= 1
        else:
            # if chosen level full, try others
            available = [i for i in range(height) if len(levels[i]) < width]
            if not available:
                break
            lvl = rng.choice(available)
            levels[lvl].append(node_id)
            node_id += 1
            remaining -= 1
    return levels

def add_minimum_edges(levels: List[List[int]], rng=random):
    edges = set()
    height = len(levels)
    # for each level >0, connect each node to a random node in previous level
    for lvl in range(1, height):
        for node in levels[lvl]:
            parent = rng.choice(levels[lvl-1])
            edges.add((parent, node))
    return edges

def add_extra_edges_between_levels(levels: List[List[int]], existing_edges:set, edge_prob:float, rng=random):
    # add random edges between consecutive levels with probability edge_prob
    height = len(levels)
    edges = set(existing_edges)
    for lvl in range(0, height-1):
        for u in levels[lvl]:
            for v in levels[lvl+1]:
                if (u,v) in edges:
                    continue
                if rng.random() < edge_prob:
                    edges.add((u,v))
    return edges

def assign_node_loads(n:int, max_cl_mi:int, rng=random):
    # CL_i in MI, uniformly random [1, max_cl_mi]
    loads = {i: rng.randint(1, max_cl_mi) for i in range(1, n+1)}
    return loads

def compute_avg_edge_weight_for_ccr(loads:Dict[int,int], edges:set, CCR:float, B_mb_s:float, Cb_mips:float):
    # Derive average edge weight (MB) so that:
    # CCR = (avg_comm_time) / (avg_comp_time)
    # avg_comm_time = avg_edge_weight (MB) / B_mb_s  (seconds)
    # avg_comp_time = (avg_CL (MI) / Cb_mips) (seconds)
    # => avg_edge_weight = CCR * B_mb_s * (avg_CL / Cb_mips)
    n = len(loads)
    if n == 0 or Cb_mips <= 0:
        return 0.0
    avg_cl_mi = sum(loads.values()) / n
    avg_edge_mb = CCR * B_mb_s * (avg_cl_mi / Cb_mips)
    # total MB to distribute across edges:
    total_mb = avg_edge_mb * max(1, len(edges))
    return avg_edge_mb, total_mb

def distribute_edge_weights(edges:set, total_mb:float, rng=random):
    # distribute total_mb across edges as random positive integers MB (at least 0.1 MB)
    edge_list = list(edges)
    m = len(edge_list)
    if m == 0:
        return {}
    # give each edge at least tiny amount to avoid zeros
    remaining = total_mb
    weights = {}
    # random partition via Dirichlet-like allocation
    rand_vals = [rng.random() for _ in range(m)]
    s = sum(rand_vals)
    for i, e in enumerate(edge_list):
        fraction = rand_vals[i] / s if s>0 else 1.0/m
        mb = max(0.001, fraction * total_mb)  # floor to 1 KB (0.001 MB) to avoid zero
        weights[e] = mb * 1e6  # convert MB -> bytes
    return weights

# -------------------------
# API to produce the workflow dict
# -------------------------

def generate_random_workflow(
    n:int = 10,
    alpha:float = 0.5,
    edge_prob:float = 0.3,
    Cb_mips:float = 100.0,
    B_mb_s:float = 10.0,
    CCR:float = 0.5,
    max_cl_mi:int = 50,
    seed:int = 42
):
    """
    Generates a workflow dict similar to the paper's Algorithm 1.
    Returns dict:
    {
      "tasks": {id: {"v": <MI*1e6 or same units as your code>}, ...},
      "edges": {(u,v): bytes, ...},
      "N": n
    }
    Units:
      - tasks 'v' are returned in MI * 1e6 (so matches your example e.g. 2e6)
      - edges are bytes (MB * 1e6)
    """
    rng = random.Random(seed)
    height, width = compute_height_width(n, alpha)
    levels = distribute_nodes_across_levels(n, height, width, rng=rng)
    edges_min = add_minimum_edges(levels, rng=rng)
    edges_all = add_extra_edges_between_levels(levels, edges_min, edge_prob, rng=rng)

    loads = assign_node_loads(n, max_cl_mi, rng=rng)  # in MI
    avg_edge_mb, total_mb = compute_avg_edge_weight_for_ccr(loads, edges_all, CCR, B_mb_s, Cb_mips)
    edge_weights = distribute_edge_weights(edges_all, total_mb, rng=rng)

    # build workflow dict (match your expected format)
    tasks = {}
    for i in range(1, n+1):
        # convert MI -> number consistent with your sample 'v' values (they used e6)
        tasks[i] = {"v": float(loads[i]) * 1e6}  # 1 MI => 1e6 units

    edges = {}
    for e, bytes_val in edge_weights.items():
        edges[tuple(e)] = float(bytes_val)

    workflow = {
        "tasks": tasks,
        "edges": edges,
        "N": n,
        "meta": {
            "height": height,
            "width": width,
            "edge_prob": edge_prob,
            "CCR": CCR,
            "avg_edge_MB": avg_edge_mb
        }
    }
    return workflow

def serialize_workflow_for_json(wf: dict) -> dict:
    """
    Convert workflow dict (which may use tuple keys for edges and int keys for tasks)
    into a JSON-serializable dict:
      - tasks: keys converted to strings (so JSON has string keys)
      - edges: converted to a list of {"u":int,"v":int,"bytes":float}
    This keeps semantics clear and is robust across languages.
    """
    tasks = {}
    for k, v in wf.get("tasks", {}).items():
        # ensure keys are strings in JSON, but preserve original numeric value in output if needed
        tasks[str(k)] = v

    edges_raw = wf.get("edges", {})
    edges_list = []
    # edges_raw keys may be tuples (u,v) or string keys like "u,v"
    for k, val in edges_raw.items():
        if isinstance(k, tuple):
            u, v = k
        elif isinstance(k, str):
            # try to parse "u,v" or "u v"
            if "," in k:
                u_str, v_str = k.split(",", 1)
            elif " " in k:
                u_str, v_str = k.split(" ", 1)
            elif "-" in k:
                u_str, v_str = k.split("-", 1)
            else:
                # fallback: put whole string as u and v = None
                u_str, v_str = k, ""
            try:
                u = int(u_str)
                v = int(v_str)
            except:
                u = u_str
                v = v_str
        else:
            # fallback: stringify key
            u = str(k)
            v = None

        edges_list.append({"u": u, "v": v, "bytes": float(val)})

    out = {
        "tasks": tasks,
        "edges": edges_list,
        "N": wf.get("N"),
        "meta": wf.get("meta", {})
    }
    return out

def deserialize_workflow_from_json(saved: dict) -> dict:
    """
    Convert JSON-loaded workflow back to in-memory shape:
    - tasks keys -> int
    - edges list -> dict keyed by (u,v) tuples
    """
    tasks_in = {}
    for k, v in saved.get("tasks", {}).items():
        try:
            tasks_in[int(k)] = v
        except:
            tasks_in[k] = v

    edges_in = {}
    for e in saved.get("edges", []):
        u = e.get("u")
        v = e.get("v")
        try:
            u_i = int(u)
            v_i = int(v)
            edges_in[(u_i, v_i)] = float(e.get("bytes", 0.0))
        except:
            edges_in[(u, v)] = float(e.get("bytes", 0.0))

    return {"tasks": tasks_in, "edges": edges_in, "N": saved.get("N"), "meta": saved.get("meta", {})}


def serialize_env_for_json(env: dict) -> dict:
    """
    Convert environment dict to JSON-friendly format.
    - locations: kept the same (keys should be primitives)
    - DR: dict with tuple keys -> list of {"u":int,"v":int,"value":float}
    - DE/VR/VE expected to have primitive keys (int -> float) so they are preserved
    """
    out = {}
    out["locations"] = env.get("locations", {})

    # handle DR map (tuple keys)
    DR = env.get("DR", {})
    dr_list = []
    for k, v in DR.items():
        if isinstance(k, tuple):
            u, w = k
        elif isinstance(k, str):
            # try parse "u,v"
            if "," in k:
                a, b = k.split(",", 1)
                try:
                    u, w = int(a), int(b)
                except:
                    u, w = a, b
            else:
                u, w = k, None
        else:
            u, w = str(k), None
        dr_list.append({"u": u, "v": w, "value": float(v)})
    out["DR"] = dr_list

    # preserve the other maps as-is (they should have primitive keys)
    out["DE"] = env.get("DE", {})
    out["VR"] = env.get("VR", {})
    out["VE"] = env.get("VE", {})
    return out

# -------------------------
# Environment builder
# -------------------------
def create_environment_dict(locations_types:Dict[int,str],
                            DR_map:Dict[Tuple[int,int],float],
                            DE_map:Dict[int,float],
                            VR_map:Dict[int,float],
                            VE_map:Dict[int,float]):
    """
    Package environment dictionaries exactly as your snippet expects.
    DR_map values: time per byte (seconds/byte or ms/byte as you want). Keep consistent.
    DE_map: energy per byte (J/byte or mJ/byte depending on your convention)
    VR_map: time per cycle (s/cycle or ms/cycle)
    VE_map: energy per cycle
    """
    return {
        "locations": locations_types,
        "DR": DR_map,
        "DE": DE_map,
        "VR": VR_map,
        "VE": VE_map
    }

# -------------------------
# Example main to produce files (customize params here)
# -------------------------

if __name__ == "__main__":
    # Example generation parameters (you can change these)
    n = 6
    alpha = 0.7
    edge_prob = 0.4
    Cb_mips = 100.0    # base computational capacity (MIPS)
    B_mb_s = 10.0      # bandwidth reference in MB/s
    CCR = 0.2
    max_cl_mi = 40
    seed = 12345

    # Generate workflow
    wf = generate_random_workflow(
        n=n,
        alpha=alpha,
        edge_prob=edge_prob,
        Cb_mips=Cb_mips,
        B_mb_s=B_mb_s,
        CCR=CCR,
        max_cl_mi=max_cl_mi,
        seed=seed
    )

    # Example environment maps (you can replace / tune with your values)
    locations_types = {0: "iot", 1: "edge_a", 2: "edge_b", 3: "cloud"}
    # DR: time per byte (seconds/byte) — keep consistent with your usage (ms/byte or s/byte)
    DR_map = {
        (0,0):0.0, (1,1):0.0, (2,2):0.0, (3,3):0.0,
        (0,1):1.0e-05, (1,0):1.0e-05,
        (0,2):1.5e-05, (2,0):1.5e-05,
        (0,3):2.0e-03, (3,0):2.0e-03,
        (1,2):4.0e-05, (2,1):4.0e-05,
        (1,3):6.0e-05, (3,1):6.0e-05,
        (2,3):3.0e-05, (3,2):3.0e-05,
    }

    DE_map = {0: 1.20e-4, 1: 2.50e-5, 2: 2.20e-5, 3: 1.80e-5}
    VR_map = {0: 1.0e-7, 1: 3.0e-8, 2: 2.0e-8, 3: 1.0e-8}
    VE_map = {0: 6.0e-7, 1: 3.0e-7, 2: 2.0e-7, 3: 1.2e-7}

    env = create_environment_dict(
        locations_types=locations_types,
        DR_map=DR_map,
        DE_map=DE_map,
        VR_map=VR_map,
        VE_map=VE_map
    )

    with open("workflow.json", "w") as f:
        json.dump(serialize_workflow_for_json(wf), f, indent=2)

    with open("env.json", "w") as f:
        json.dump(serialize_env_for_json(env), f, indent=2)

    print("Wrote workflow.json and env.json")
