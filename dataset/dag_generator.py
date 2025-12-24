#!/usr/bin/env python3
"""
generate_dataset_json_fixed_env.py

Generates workflow DAG instances and environment matrices:
 - DR_map: full n x n matrix (in-memory as dict[(i,j)] -> float)
 - DE_map, VR_map, VE_map: per-location dicts (vectors)
Saves JSON with JSON-safe representations (DR keys as "i,j" strings;
 edges listed as objects).
"""

import argparse
import json
import math
import random
import uuid
from typing import List, Dict, Any, Tuple

def build_location_types(num_remote: int) -> Dict[int, str]:
    """
    Randomly assign device types to remote locations.
    Ensures:
        - 0 → 'iot' (fixed)
        - remote devices 1..num_remote are randomly 'edge' or 'cloud'
        - (optional) ensure at least one cloud if num_remote > 0
    """
    rng = random.Random()

    locs = {0: "iot"}

    if num_remote == 0:
        return locs

    # Randomly assign each remote device
    remote_types = []
    for _ in range(num_remote):
        remote_types.append(rng.choice(["edge", "cloud"]))

    # Ensure at least one cloud exists (optional, but recommended)
    if "cloud" not in remote_types:
        idx = rng.randint(0, num_remote - 1)
        remote_types[idx] = "cloud"

    # Store into mapping
    for i in range(1, num_remote + 1):
        locs[i] = remote_types[i - 1]

    return locs


def make_instance_smallvalues(
    v: int,
    edge_prob: float,
    num_remote: int,
    seed: int
) -> Dict[str, Any]:
    """
    Create a single instance:
      - tasks: dict[int] -> {"v": cycles}
      - edges: dict[(u,v)] -> bytes (in-memory). For JSON we emit list entries.
      - location_types: {loc_id: type}
      - env maps:
          DR_map: dict[(i,j)] -> float (ms/byte)
          DE_map, VR_map, VE_map: dict[loc] -> float
    """
    rng = random.Random(seed)

    # --- DAG topology generation (layered) ---
    alpha = 1.0
    height = max(1, int(math.ceil(math.sqrt(v) / alpha)))
    width = max(1, int(math.ceil(math.sqrt(v) * alpha)))

    # fill grid with unique task ids 1..v
    grid = [[-1 for _ in range(width)] for _ in range(height)]
    lvlcount = [0] * height
    curr = 1
    for i in range(height):
        grid[i][0] = curr
        lvlcount[i] = 1
        curr += 1
    while curr <= v:
        r = rng.randint(0, height - 1)
        if lvlcount[r] >= width:
            continue
        grid[r][lvlcount[r]] = curr
        lvlcount[r] += 1
        curr += 1

    N = v
    node_ids = [tid for row in grid for tid in row if tid != -1]

    # build parent/children temporary maps
    parents: Dict[int, List[int]] = {tid: [] for tid in node_ids}
    children: Dict[int, List[int]] = {tid: [] for tid in node_ids}

    # mandatory parent connection (connect each node to at least one parent)
    for lvl in range(1, height):
        for i in range(lvlcount[lvl]):
            node = grid[lvl][i]
            prev_cnt = lvlcount[lvl-1]
            parent = grid[lvl-1][0] if prev_cnt == 1 else grid[lvl-1][rng.randint(0, prev_cnt-1)]
            parents[node].append(parent)
            children[parent].append(node)

    # ensure every non-last-level node has at least one child
    for lvl in range(0, height-1):
        for i in range(lvlcount[lvl]):
            node = grid[lvl][i]
            if len(children[node]) == 0:
                nxt_cnt = lvlcount[lvl+1]
                chosen = grid[lvl+1][0] if nxt_cnt == 1 else grid[lvl+1][rng.randint(0, nxt_cnt-1)]
                children[node].append(chosen)
                parents[chosen].append(node)

    # extra random edges between consecutive levels
    for lvl in range(0, height-1):
        for i in range(lvlcount[lvl]):
            node = grid[lvl][i]
            for j in range(lvlcount[lvl+1]):
                child = grid[lvl+1][j]
                if (rng.random() < edge_prob) and (child not in children[node]):
                    children[node].append(child)
                    parents[child].append(node)

    # collect edges list as dict keyed by tuple (u,v) -> bytes
    edges_dict: Dict[Tuple[int,int], float] = {}
    for u, chs in children.items():
        for v_ in chs:
            # pick size clustered across small/medium/large as in your earlier script
            size = rng.choice([
                rng.uniform(0.5e6, 1.5e6),
                rng.uniform(1.5e6, 3.0e6),
                rng.uniform(3.0e6, 15.0e6)
            ])
            edges_dict[(int(u), int(v_))] = float(size)

    edgecount = len(edges_dict)

    # --- TASK LOADS (cycles) ---
    tasks: Dict[int, Dict[str, float]] = {}
    for tid in node_ids:
        # use range roughly matching your earlier example (1e6 - 35e6 cycles)
        v_i = rng.randint(1_000_000, 35_000_000)
        tasks[int(tid)] = {"v": float(v_i)}

    # --- LOCATION TYPES ---
    location_types = build_location_types(num_remote)

    # --- ENV maps (produce full n x n DR matrix) ---
    locations = list(range(0, num_remote + 1))  # 0..num_remote inclusive
    DR_map: Dict[Tuple[int,int], float] = {}
    DE_map: Dict[int, float] = {}
    VR_map: Dict[int, float] = {}
    VE_map: Dict[int, float] = {}

    # DR: data time consumption ms/byte -> full n x n
    # We'll mimic the example values you gave:
    #   - local links 0.0
    #   - IoT<->edge: small (~1e-5)
    #   - IoT<->cloud: large (~2e-3)
    #   - edge<->edge: mid (~3e-5 .. 6e-5)
    for a in locations:
        for b in locations:
            if a == b:
                DR_map[(a,b)] = 0.0
            else:
                # IoT index is 0
                if (a == 0 and (location_types.get(b, "") == "cloud")) or (b == 0 and (location_types.get(a, "") == "cloud")):
                    # IoT <-> Cloud slow path
                    DR_map[(a,b)] = float(rng.uniform(1.5e-3, 3.0e-3))  # ~0.0015 - 0.003 ms/byte
                elif a == 0 or b == 0:
                    # IoT <-> Edge (fast-ish)
                    DR_map[(a,b)] = float(rng.uniform(0.8e-5, 2.0e-5))   # ~0.8e-5 - 2.0e-5 ms/byte
                else:
                    # edge <-> edge or edge <-> cloud (inter-remote)
                    # choose in a slightly wider band
                    DR_map[(a,b)] = float(rng.uniform(3.0e-5, 6.0e-5))

    # DE (mJ/byte) per loc - use example values
    for l in locations:
        if l == 0:
            DE_map[l] = 1.20e-4
        else:
            # remote nodes: slight variations around example
            DE_map[l] = float(rng.uniform(1.8e-5, 2.5e-5))

    # VR (ms/cycle) per loc - example: IoT slower
    for l in locations:
        if l == 0:
            VR_map[l] = 1.0e-7
        else:
            VR_map[l] = float(rng.uniform(1.0e-8, 4.0e-8))

    # VE (mJ/cycle)
    for l in locations:
        if l == 0:
            VE_map[l] = 6.0e-7
        else:
            VE_map[l] = float(rng.uniform(1.2e-7, 3.0e-7))

    costs = {"CT": 0.2, "CE": 1.20}
    mode = {"delta_t": 1, "delta_e": 1}

    meta = {
        "seed": seed,
        "v": v,
        "edge_prob": edge_prob,
        "num_remote": num_remote,
        "edgecount": edgecount
    }

    # In-memory instance uses tuple keys for DR_map and edge dict for correctness.
    instance_in_memory = {
        "workflow": {
            "tasks": tasks,              # int-keyed dict
            "edges": edges_dict,         # tuple-keyed dict in-memory
            "N": N
        },
        "location_types": location_types,
        "env": {
            "DR": DR_map,               # tuple-keyed dict in-memory (n x n)
            "DE": DE_map,
            "VR": VR_map,
            "VE": VE_map
        },
        "costs": costs,
        "mode": mode,
        "meta": meta
    }
    return instance_in_memory

# -------------------------
# JSON serialization helpers
# -------------------------
def serialize_instance_for_json(inst: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert an in-memory instance (tuple keys etc.) into a JSON-serializable object.
    - edges (tuple keys) -> list of {"u": int, "v": int, "bytes": float}
    - DR (tuple keys) -> dict with string keys "u,v" -> float (or list rows; both work)
    - DE/VR/VE: keep as dict (note JSON will turn numeric keys to strings)
    """
    wf = inst["workflow"]
    tasks = {str(k): v for k, v in wf["tasks"].items()}  # keys -> strings (JSON)
    edges_list = []
    edges_src = wf["edges"]
    # edges_src may be dict with tuple keys
    for k, bytes_val in edges_src.items():
        if isinstance(k, tuple):
            u, v = k
        else:
            # if key already string like "1,2", parse it
            if isinstance(k, str) and "," in k:
                u_s, v_s = k.split(",", 1)
                try:
                    u, v = int(u_s), int(v_s)
                except:
                    u, v = u_s, v_s
            else:
                u, v = k, None
        edges_list.append({"u": int(u), "v": int(v), "bytes": float(bytes_val)}) # type: ignore

    # DR map: convert (i,j) tuple keys to "i,j" strings for JSON safety
    DR_src = inst["env"]["DR"]
    DR_json = {}
    for k, v in DR_src.items():
        if isinstance(k, tuple):
            key = f"{k[0]},{k[1]}"
        else:
            key = str(k)
        DR_json[key] = float(v)

    # DE/VR/VE are dicts keyed by ints; JSON will stringify keys.
    DE_json = {str(k): float(v) for k, v in inst["env"]["DE"].items()}
    VR_json = {str(k): float(v) for k, v in inst["env"]["VR"].items()}
    VE_json = {str(k): float(v) for k, v in inst["env"]["VE"].items()}

    out = {
        "workflow": {"tasks": tasks, "edges": edges_list, "N": wf.get("N")},
        "location_types": {str(k): str(v) for k, v in inst["location_types"].items()},
        "env": {"DR": DR_json, "DE": DE_json, "VR": VR_json, "VE": VE_json},
        "costs": inst.get("costs", {}),
        "mode": inst.get("mode", {}),
        "meta": inst.get("meta", {})
    }
    return out

def deserialize_env_from_json(env_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert JSON env representation back into in-memory maps:
      - DR keys "i,j" -> tuple (i,j)
      - DE/VR/VE keys string -> int
    """
    DR = {}
    for k, v in env_json.get("DR", {}).items():
        if isinstance(k, str) and "," in k:
            a, b = k.split(",", 1)
            try:
                ai, bi = int(a), int(b)
            except:
                ai, bi = a, b
            DR[(ai, bi)] = float(v)
        else:
            DR[(k,)] = float(v)

    DE = {int(k): float(v) for k, v in env_json.get("DE", {}).items()}
    VR = {int(k): float(v) for k, v in env_json.get("VR", {}).items()}
    VE = {int(k): float(v) for k, v in env_json.get("VE", {}).items()}

    return {"DR": DR, "DE": DE, "VR": VR, "VE": VE}

# -------------------------
# Dataset generation driver
# -------------------------
# -------------------------
# Dataset generation driver (UPDATED to randomize num_remote per-instance)
# -------------------------
def generate_dataset(
    out_file: str,
    count: int,
    min_v: int,
    max_v: int,
    edge_prob: float,
    min_remote: int,
    max_remote: int,
    seed: int
):
    """
    Generates `count` instances. For each instance, randomly pick num_remote in [min_remote, max_remote].
    Each instance will therefore have locations 0..num_remote inclusive (0 == IoT).
    """
    rng = random.Random(seed)
    dataset = []
    for i in range(count):
        v = rng.randint(min_v, max_v)
        s = seed + i
        # pick num_remote per-instance (at least 0)
        num_remote = rng.randint(min_remote, max_remote)
        inst = make_instance_smallvalues(v=v, edge_prob=edge_prob, num_remote=num_remote, seed=s)
        # add primary keys and meta
        inst_id = str(uuid.uuid4())
        inst_pk = i
        inst["id"] = inst_id
        inst["pk"] = inst_pk
        inst["meta"]["id"] = inst_id
        inst["meta"]["pk"] = inst_pk
        inst["meta"]["num_remote"] = num_remote

        # serialize for JSON (keeps DR as "i,j" keys and edges as list)
        dataset.append(serialize_instance_for_json(inst))

    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(dataset, fh, indent=2)
    print(f"Wrote {len(dataset)} instances to {out_file}")


# -------------------------
# CLI entrypoint
# -------------------------
# -------------------------
# CLI entrypoint (single, unified)
# -------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate DAG dataset. Use either --num_remote (fixed per dataset) "
                    "or --min_remote/--max_remote (randomize per-instance)."
    )
    p.add_argument("--out", type=str, default="dataset.json")
    p.add_argument("--count", type=int, default=50)
    p.add_argument("--min_v", type=int, default=6)
    p.add_argument("--max_v", type=int, default=7)
    p.add_argument("--edge_prob", type=float, default=0.25)

    # Two mutually-supporting ways to specify remotes:
    p.add_argument("--num_remote", type=int, default=None,
                   help="(optional) FIXED number of remote locations for all instances. "
                        "If provided, overrides min_remote/max_remote (legacy mode).")
    p.add_argument("--min_remote", type=int, default=1,
                   help="Minimum number of remote locations (used when --num_remote not provided).")
    p.add_argument("--max_remote", type=int, default=8,
                   help="Maximum number of remote locations (used when --num_remote not provided).")

    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Validate and normalize remote arguments
    if args.num_remote is not None:
        if args.num_remote < 0:
            raise SystemExit("num_remote must be >= 0")
        # Use the legacy fixed-num mode: set min_remote == max_remote == num_remote
        min_remote = max_remote = args.num_remote
        print(f"[INFO] Running in fixed-num mode: num_remote = {args.num_remote}")
    else:
        min_remote = args.min_remote
        max_remote = args.max_remote
        if min_remote < 0:
            raise SystemExit("min_remote must be >= 0")
        if max_remote < min_remote:
            raise SystemExit("max_remote must be >= min_remote")
        print(f"[INFO] Running in randomized mode: min_remote = {min_remote}, max_remote = {max_remote}")

    generate_dataset(
        out_file=args.out,
        count=args.count,
        min_v=args.min_v,
        max_v=args.max_v,
        edge_prob=args.edge_prob,
        min_remote=min_remote,
        max_remote=max_remote,
        seed=args.seed
    )
