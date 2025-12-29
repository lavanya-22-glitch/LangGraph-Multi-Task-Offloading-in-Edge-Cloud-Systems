# main.py - UPDATED: Integrated memory system for learning across experiments
import os, json, dotenv, csv
from datetime import datetime
from langgraph.graph import StateGraph, END, START
from agents.planner import PlannerAgent
from agents.evaluator import EvaluatorAgent
from agents.output import OutputAgent
from typing import TypedDict, Optional, List, Dict, Tuple, Any
from core.workflow import Workflow
from core.environment import Environment
from core.memory_manager import WorkflowMemory

dotenv.load_dotenv()
# Option A: Hardcode it
GEMINI_API_KEY = "your api key here"

# OR Option B: If you have a .env file
# GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

class AgenticState(TypedDict, total=False):
    query: str
    env: dict
    workflow: dict          
    params: Optional[dict]  
    plan: Optional[str]
    evaluation: Optional[str]
    output: Optional[dict]
    optimal_policy: Optional[List[int]]
    experiment_id: Optional[str]
    memory_manager: Optional[WorkflowMemory]

def initialize_log_file(log_file: str, state_data: dict):
    """Initialize the log file with header and environment/workflow details."""
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MULTI-AGENT TASK OFFLOADING OPTIMIZATION - EXECUTION TRACE\n")
        f.write("="*80 + "\n")
        f.write(f"Execution Time: {datetime.now().isoformat()}\n")
        f.write("="*80 + "\n\n")
        
        # Log environment details
        f.write("="*80 + "\n")
        f.write("ENVIRONMENT CONFIGURATION\n")
        f.write("="*80 + "\n")
        env = state_data.get('env', {})
        
        # Network topology
        dr = env.get('DR', {})
        if dr:
            f.write("\nNetwork Data Time Consumption (DR - ms/byte):\n")
            f.write("-" * 40 + "\n")
            for key, rate in sorted(dr.items()):
                if isinstance(key, tuple):
                    src, dst = key
                    f.write(f"  Link ({src} → {dst}): {rate:.6f} ms/byte\n")
        
        # Data energy coefficients
        de = env.get('DE', {})
        if de:
            f.write("\nData Energy Consumption (DE - mJ/byte):\n")
            f.write("-" * 40 + "\n")
            for loc, coeff in sorted(de.items()):
                f.write(f"  Location {loc}: {coeff:.6f} mJ/byte\n")
        
        # Task time consumption
        vr = env.get('VR', {})
        if vr:
            f.write("\nTask Time Consumption (VR - ms/cycle):\n")
            f.write("-" * 40 + "\n")
            for loc, rate in sorted(vr.items()):
                f.write(f"  Location {loc}: {rate:.6e} ms/cycle\n")
        
        # Task energy consumption
        ve = env.get('VE', {})
        if ve:
            f.write("\nTask Energy Consumption (VE - mJ/cycle):\n")
            f.write("-" * 40 + "\n")
            for loc, energy in sorted(ve.items()):
                f.write(f"  Location {loc}: {energy:.6e} mJ/cycle\n")
        
        # Parameters
        params = state_data.get('params', {})
        if params:
            f.write("\nOptimization Parameters:\n")
            f.write("-" * 40 + "\n")
            for key, value in params.items():
                f.write(f"  {key}: {value}\n")
        
        f.write("\n" + "="*80 + "\n\n")
        
        # Log workflow details
        f.write("="*80 + "\n")
        f.write("WORKFLOW CONFIGURATION\n")
        f.write("="*80 + "\n")
        workflow = state_data.get('workflow', {})
        tasks = workflow.get('tasks', {})
        edges = workflow.get('edges', {})
        N = workflow.get('N', 0)
        
        f.write(f"\nTotal Real Tasks (N): {N}\n")
        f.write("-" * 40 + "\n")
        
        for task_id in sorted(tasks.keys()):
            task_data = tasks[task_id]
            size = task_data.get('v', 0)
            
            f.write(f"\nTask {task_id}:\n")
            f.write(f"  CPU Cycles (v_{task_id}): {size:.2e} cycles\n")
            
            # Find dependencies from edges
            deps = {j: d for (i, j), d in edges.items() if i == task_id}
            if deps:
                f.write(f"  Dependencies:\n")
                for dep_id, data_size in sorted(deps.items()):
                    f.write(f"    → Task {dep_id}: {data_size:.2e} bytes\n")
            else:
                f.write(f"  Dependencies: None\n")
        
        f.write("\n" + "="*80 + "\n\n")
        f.write("="*80 + "\n")
        f.write("AGENT INTERACTIONS\n")
        f.write("="*80 + "\n\n")

def build_agentic_workflow(log_file: str = "agent_trace.txt", memory_manager: WorkflowMemory = None):
    workflow = StateGraph(AgenticState)

    planner = PlannerAgent(GEMINI_API_KEY, log_file=log_file, memory_manager=memory_manager) 
    evaluator = EvaluatorAgent(GEMINI_API_KEY, log_file=log_file)
    output = OutputAgent(GEMINI_API_KEY, log_file=log_file)

    # Add nodes
    workflow.add_node("planner", planner.run)
    workflow.add_node("evaluator", evaluator.run)
    workflow.add_node("output", output.run)

    # Define edges
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "evaluator")
    workflow.add_edge("evaluator", "output")
    workflow.add_edge("output", END)

    return workflow.compile()

def run_workflow(task_description: str, state_data: dict, log_file: str = "agent_trace.txt",
                memory_manager: WorkflowMemory = None):
    """
    state_data should include:
      - env: environment parameters (as dict with DR, DE, VR, VE)
      - workflow: workflow dict (tasks, edges, N)
      - params: optional evaluator parameters (CT, CE, delta_t, delta_e)
      - experiment_id: unique identifier for this experiment
    """
    # Initialize log file with environment and workflow details
    initialize_log_file(log_file, state_data)
    
    workflow = build_agentic_workflow(log_file, memory_manager)
    
    # Add memory_manager to state so it can be used by agents
    state_data['memory_manager'] = memory_manager
    
    result = workflow.invoke({
        "query": task_description,
        **state_data  
    })

    # Add summary to log file
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "="*80 + "\n")
        f.write("EXECUTION SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Query: {task_description}\n")
        f.write(f"Optimal Policy: {result.get('optimal_policy', [])}\n")
        f.write(f"Evaluation: {result.get('evaluation', 'N/A')}\n")
        f.write("\n" + "="*80 + "\n")
        f.write("END OF TRACE\n")
        f.write("="*80 + "\n")
    
    print(f"\n✅ Complete execution trace saved to: {log_file}")
    
    return result

def create_environment_dict(
    locations_types: Dict[int, str],
    DR_map: Dict[Tuple[int, int], float],
    DE_map: Dict[int, float],
    VR_map: Dict[int, float],
    VE_map: Dict[int, float]
) -> dict:
    """
    Create environment dictionary in the expected format.
    
    Args:
        locations_types: {location_id: type} where type in {'iot', 'edge', 'cloud'}
        DR_map: {(li, lj): ms/byte}
        DE_map: {l: mJ/byte}
        VR_map: {l: ms/cycle}
        VE_map: {l: mJ/cycle}
    
    Returns:
        Dictionary with DR, DE, VR, VE maps
    """
    return {
        "locations": locations_types,
        "DR": DR_map,
        "DE": DE_map,
        "VR": VR_map,
        "VE": VE_map
    }

def save_results_csv(out_dir: str, workflow_meta: dict, workflow_dict: dict,
                     placement_policy: dict, optimal_cost: float, metrics: dict = None):
    """
    Create a CSV file per workflow summarizing results.

    - out_dir: directory to save CSVs
    - workflow_meta: metadata dict (e.g., meta.pk or meta.id or seed)
    - workflow_dict: as returned by parse_dataset_object (contains 'tasks' and 'edges')
    - placement_policy: mapping task_id -> assigned location (int or str)
    - optimal_cost: numeric cost for this placement
    - metrics: optional dict of other metrics (latency, energy, etc.)
    """
    os.makedirs(out_dir, exist_ok=True)
    # choose filename based on meta if present
    w_id = workflow_meta.get('id') or workflow_meta.get('pk') or workflow_meta.get('seed') or 'unknown'
    fname = f"workflow_{w_id}.csv"
    path = os.path.join(out_dir, fname)

    tasks = workflow_dict.get('tasks', {})
    edges = workflow_dict.get('edges', {})

    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # summary section
        writer.writerow(['SUMMARY'])
        writer.writerow(['workflow_id', w_id])
        writer.writerow(['timestamp', datetime.utcnow().isoformat() + 'Z'])
        writer.writerow(['num_tasks', len(tasks)])
        writer.writerow(['num_edges', len(edges)])
        writer.writerow(['optimal_cost', optimal_cost])
        # write additional metrics if any
        if metrics:
            for k, v in metrics.items():
                writer.writerow([k, v])
        writer.writerow([])

        # placement policy section
        writer.writerow(['PLACEMENT POLICY'])
        writer.writerow(['task_id', 'task_v (if available)', 'assigned_location'])
        for tid in sorted(tasks, key=lambda x: int(x)):
            tval = tasks[tid].get('v') if isinstance(tasks[tid], dict) else ''
            assigned = placement_policy.get(int(tid)) if placement_policy is not None else ''
            writer.writerow([tid, tval, assigned])
        writer.writerow([])

        # edges / bytes (optional, useful for benchmarking)
        writer.writerow(['EDGES'])
        writer.writerow(['u', 'v', 'bytes'])
        for (u, v), b in edges.items():
            writer.writerow([u, v, b])

    # returns the path for convenience
    return path

def parse_dataset_object(dataset_obj: dict) -> Tuple[dict, dict, dict, dict]:
    """
    Parse a dataset object (robust to the dataset.json format used in uploads).
    Returns (workflow_dict, locations_types, env_dict, params)
    """
    # -------------------- WORKFLOW --------------------
    workflow_data = dataset_obj['workflow']

    # tasks: keys may be strings -> convert to int
    tasks = {int(k): {"v": float(v["v"])} for k, v in workflow_data['tasks'].items()}

    # edges: accept either list-of-dicts {"u":..,"v":..,"bytes":..} OR list-of-lists [u,v,bytes]
    edges_raw = workflow_data.get('edges', [])
    edges = {}
    if isinstance(edges_raw, dict):
        # in case someone provided a dict keyed by "u,v" (unlikely) handle gracefully
        for k, val in edges_raw.items():
            if isinstance(val, dict) and 'bytes' in val:
                # attempt to split key
                try:
                    u_str, v_str = k.split(',')
                    u, v = int(u_str), int(v_str)
                    edges[(u, v)] = float(val['bytes'])
                except Exception:
                    continue
    else:
        for e in edges_raw:
            if isinstance(e, dict):
                u = int(e.get('u'))
                v = int(e.get('v'))
                b = float(e.get('bytes', e.get('bw', 0.0)))
                edges[(u, v)] = b
            elif isinstance(e, (list, tuple)) and len(e) >= 3:
                edges[(int(e[0]), int(e[1]))] = float(e[2])
            else:
                # skip unknown formats
                continue

    workflow_dict = {"tasks": tasks, "edges": edges, "N": int(workflow_data.get('N', len(tasks)))}

    # -------------------- LOCATION TYPES --------------------
    # dataset may have: {"0": "iot", "1":"edge", ...} OR {"0": 0, "1":1, ...}
    raw_loc_types = dataset_obj.get('location_types', {})
    # attempt to detect form
    locations_types = {}
    for k, v in raw_loc_types.items():
        loc_id = int(k)
        if isinstance(v, str):
            # assume already "iot"/"edge"/"cloud"
            locations_types[loc_id] = v
        else:
            # numeric codes: 0=iot,1=edge,2=cloud
            type_map = {0: "iot", 1: "edge", 2: "cloud"}
            locations_types[loc_id] = type_map.get(int(v), "edge")

    # ensure location 0 exists and is iot (warn if we auto-fix)
    if 0 not in locations_types:
        locations_types[0] = "iot"
        print("⚠️  Warning: Location 0 (IoT device) missing in dataset. Added as 'iot'.")
    elif locations_types[0] != "iot":
        print(f"⚠️  Warning: Location 0 was type '{locations_types[0]}'. Forcing to 'iot'.")
        locations_types[0] = "iot"

    # -------------------- ENVIRONMENT --------------------
    # dataset env commonly has:
    # "DR": { "0,1": rate, "1,0": rate, ... }
    # "DE","VR","VE": { "0": val, "1": val, ... }
    env_raw = dataset_obj.get('env', {})

    # Parse DR
    DR_map = {}
    dr_raw = env_raw.get('DR', {})
    if isinstance(dr_raw, dict):
        for k, v in dr_raw.items():
            # key might be "i,j" or integer index string
            if isinstance(k, str) and ',' in k:
                i_str, j_str = k.split(',')
                i, j = int(i_str.strip()), int(j_str.strip())
                DR_map[(i, j)] = float(v)
            else:
                # fallback: skip or attempt conversion
                try:
                    idx = int(k)
                    DR_map[(idx, idx)] = float(v)
                except Exception:
                    continue
    else:
        # assume list-of-lists like [[i,j,rate],...]
        for entry in dr_raw:
            if isinstance(entry, (list, tuple)) and len(entry) >= 3:
                DR_map[(int(entry[0]), int(entry[1]))] = float(entry[2])

    # Parse DE / VR / VE
    def _parse_scalar_map(raw_map):
        out = {}
        if isinstance(raw_map, dict):
            for k, v in raw_map.items():
                try:
                    out[int(k)] = float(v)
                except:
                    # if keys are numeric types already
                    try:
                        out[int(k)] = float(v)
                    except:
                        continue
        elif isinstance(raw_map, list):
            # list of [type_or_loc, value] entries
            for entry in raw_map:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    out[int(entry[0])] = float(entry[1])
        return out

    DE_map = _parse_scalar_map(env_raw.get('DE', {}))
    VR_map = _parse_scalar_map(env_raw.get('VR', {}))
    VE_map = _parse_scalar_map(env_raw.get('VE', {}))

    # If DE/VR/VE were provided per-location-type (0=iot,1=edge,2=cloud),
    # map them to individual locations using locations_types
    # Detect if keys look like types (0/1/2) but locations use more ids
    if all(k in (0,1,2) or k < 3 for k in DE_map.keys()) and max(locations_types.keys(), default=0) > 3:
        # expand
        DE_map_expanded = {}
        VR_map_expanded = {}
        VE_map_expanded = {}
        type_to_num = {"iot":0, "edge":1, "cloud":2}
        for loc, loc_type in locations_types.items():
            tnum = type_to_num.get(loc_type, 1)
            if tnum in DE_map: DE_map_expanded[loc] = DE_map[tnum]
            if tnum in VR_map: VR_map_expanded[loc] = VR_map[tnum]
            if tnum in VE_map: VE_map_expanded[loc] = VE_map[tnum]
        DE_map = DE_map_expanded or DE_map
        VR_map = VR_map_expanded or VR_map
        VE_map = VE_map_expanded or VE_map

    env_dict = {
        "locations": locations_types,
        "DR": DR_map,
        "DE": DE_map,
        "VR": VR_map,
        "VE": VE_map
    }

    # -------------------- PARAMS --------------------
    costs = dataset_obj.get('costs', {})
    mode = dataset_obj.get('mode', {})
    params = {
        "CT": float(costs.get('CT', 0.2)),
        "CE": float(costs.get('CE', 1.2)),
        "delta_t": int(mode.get('delta_t', 1)),
        "delta_e": int(mode.get('delta_e', 1))
    }

    return workflow_dict, locations_types, env_dict, params

def load_dataset(json_file: str = "dataset/dataset.json") -> List[dict]:
    """Load all dataset objects from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)

def calculate_experiment(dataset_obj: dict, experiment_index: int, memory_manager: WorkflowMemory):
    """
    Calculate (evaluate) a single dataset object with memory integration.
    """
    experiment_id = dataset_obj.get('id', f'exp_{experiment_index}')
    
    print(f"\n{'='*80}")
    print(f"Running experiment {experiment_index} (ID: {experiment_id})")
    print(f"{'='*80}")
    print(f"Seed: {dataset_obj['meta']['seed']}")
    print(f"Tasks: {dataset_obj['meta']['v']}, Edges: {dataset_obj['meta']['edgecount']}\n")

    # PARSE DATASET OBJECT
    workflow_dict, locations_types, env_dict, params = parse_dataset_object(dataset_obj)
    
    # Create Workflow object
    wf = Workflow.from_experiment_dict(workflow_dict)
    
    # Create Environment object
    env = Environment.from_matrices(
        types=locations_types,
        DR_matrix=env_dict["DR"],
        DE_vector=env_dict["DE"],
        VR_vector=env_dict["VR"],
        VE_vector=env_dict["VE"]
    )

    # RUN AGENTIC WORKFLOW
    log_file = f"agent_trace_exp_{experiment_index}.txt"

    result = run_workflow(
        "Find optimal offloading policy for this edge-cloud task offloading scenario", 
        {
            "env": env_dict,
            "workflow": wf.to_experiment_dict(),
            "params": params,
            "experiment_id": experiment_id
        },
        log_file=log_file,
        memory_manager=memory_manager
    )

    # <<< CSV WRITE: START >>>
    # Normalize optimal_policy into mapping task_id -> location for CSV writer
    try:
        raw_policy = result.get("optimal_policy", []) or result.get("recommended_policy", {}) or []
        placement_map = {}
        if isinstance(raw_policy, dict):
            # keys may be strings or ints
            for k, v in raw_policy.items():
                try:
                    placement_map[int(k)] = int(v)
                except Exception:
                    # keep raw if cannot cast
                    placement_map[k] = v
        elif isinstance(raw_policy, (list, tuple)):
            # convert list [l1, l2, ...] -> {1: l1, 2: l2, ...}
            for i, loc in enumerate(raw_policy, start=1):
                try:
                    placement_map[i] = int(loc)
                except Exception:
                    placement_map[i] = loc
        else:
            # unknown policy type - leave empty
            placement_map = {}

        # Determine best/optimal cost if available (from result or parsed evaluation)
        # Prefer numeric fields inside result['evaluation'] or fallback to parsed text later
        optimal_cost = None
        eval_obj = result.get("evaluation", {})
        if isinstance(eval_obj, dict):
            # common keys: 'total_cost', 'U_total', 'best_cost'
            for k in ("total_cost", "U_total", "best_cost", "cost"):
                if k in eval_obj:
                    try:
                        optimal_cost = float(eval_obj[k])
                        break
                    except Exception:
                        continue

        # fallback: if you parsed evaluation_str below, one of the parsed values will be used later;
        # but try to get a numeric best_cost from the result string too (if present).
        if optimal_cost is None:
            eval_str = result.get("evaluation", "") if isinstance(result.get("evaluation", ""), str) else result.get("evaluation", "")
            if isinstance(eval_str, str):
                import re
                m = re.search(r'U\(w,p\*\)\s*=\s*([\d.]+)', eval_str)
                if m:
                    try:
                        optimal_cost = float(m.group(1))
                    except:
                        pass

        # Prepare minimal workflow metadata for the CSV
        workflow_meta = dataset_obj.get("meta", {})
        # call save_results_csv (expects helper to be present in file)
        try:
            csv_path = save_results_csv(
                out_dir="./results_csv",
                workflow_meta=workflow_meta,
                workflow_dict=workflow_dict,
                placement_policy=placement_map,
                optimal_cost=optimal_cost,
                metrics=eval_obj if isinstance(eval_obj, dict) else {"evaluation": eval_obj}
            )
            print(f"✅ Saved CSV for experiment {experiment_id} -> {csv_path}")
        except NameError:
            # save_results_csv not defined in this file / scope
            print("⚠️ save_results_csv is not defined. Skipping CSV write. Please add the helper function.")
        except Exception as e:
            print(f"⚠️ Failed to write CSV for experiment {experiment_id}: {e}")

    except Exception as e:
        print(f"⚠️ Unexpected error while preparing CSV for experiment {experiment_id}: {e}")
    # <<< CSV WRITE: END >>>

    # SAVE TO MEMORY
    optimal_policy = result.get("optimal_policy", [])
    evaluation_str = result.get("evaluation", "")
    plan = result.get("plan", "")
    
    # Extract evaluation result details
    evaluation_result = {
        "best_policy": optimal_policy,
        "best_cost": None,
        "evaluated": 0,
        "skipped": 0
    }
    
    # Parse evaluation string to extract metrics
    import re
    cost_match = re.search(r'U\(w,p\*\)\s*=\s*([\d.]+)', evaluation_str)
    if cost_match:
        evaluation_result["best_cost"] = float(cost_match.group(1))
    
    evaluated_match = re.search(r'Evaluated:\s*(\d+)', evaluation_str)
    if evaluated_match:
        evaluation_result["evaluated"] = int(evaluated_match.group(1))
    
    skipped_match = re.search(r'Skipped:\s*(\d+)', evaluation_str)
    if skipped_match:
        evaluation_result["skipped"] = int(skipped_match.group(1))
    
    # Save execution to memory
    memory_manager.save_execution(
        workflow_dict=workflow_dict,
        env_dict=env_dict,
        params=params,
        optimal_policy=optimal_policy,
        evaluation_result=evaluation_result,
        plan=plan,
        experiment_id=experiment_id
    )

    # DISPLAY RESULTS
    print("\n" + "="*80)
    print("FINAL RESULT:")
    print("="*80)
    print(json.dumps(result.get("output", {}), indent=2))
    
    print("\n" + "="*80)
    print("OPTIMAL POLICY:")
    print("="*80)
    if optimal_policy:
        print(f"Policy vector p = {optimal_policy}")
        print("\nTask Assignments:")
        for i, location in enumerate(optimal_policy, start=1):
            loc_type = locations_types.get(location, 'unknown')
            if location == 0:
                print(f"  Task {i} → Location {location} (IoT - Local Execution)")
            else:
                print(f"  Task {i} → Location {location} ({loc_type.capitalize()} Server)")
    else:
        print("No optimal policy found.")

    print("\n" + "="*80)
    print(f"Experiment ID: {experiment_id}")
    print(f"Number of Edge Servers (E): {env.E}")
    print(f"Number of Cloud Servers (C): {env.C}")
    print(f"Total Tasks (N): {wf.N}")
    print(f"Mode: ", end="")
    if params["delta_t"] == 1 and params["delta_e"] == 1:
        print("Balanced (Time + Energy)")
    elif params["delta_t"] == 1 and params["delta_e"] == 0:
        print("Low Latency (Time Only)")
    elif params["delta_t"] == 0 and params["delta_e"] == 1:
        print("Low Power (Energy Only)")
    print("="*80)

    return result

# =============================================================================
#  HEADLESS RUNNER FOR DATA PIPELINE (With Memory Saving Enabled)
# =============================================================================

def run_workflow_headless(dataset_obj: dict):
    """
    Runs a single experiment for the pipeline_manager.
    Saves the result to 'memory_store' so the NEXT pipeline run can learn from it.
    """
    try:
        # 1. Reuse your existing parser to get clean objects
        workflow_dict, locations_types, env_dict, params = parse_dataset_object(dataset_obj)
        experiment_id = dataset_obj.get("meta", {}).get("id", "pipeline_run")

        # 2. Initialize Memory Manager
        # This allows the agent to SAVE results to disk
        memory_manager = WorkflowMemory(memory_dir="memory_store")

        # 3. Build the state
        state_data = {
            "env": env_dict,
            "workflow": workflow_dict,
            "params": params,
            "experiment_id": experiment_id,
            "memory_manager": memory_manager 
        }

        # 4. Build and Invoke Graph
        # We pass memory_manager so the agents can use it
        temp_log = "pipeline_temp.log"
        workflow = build_agentic_workflow(log_file=temp_log, memory_manager=memory_manager)
        
        result = workflow.invoke({
            "query": "Optimize offloading for data pipeline",
            **state_data  
        })

        # 5. SAVE RESULT TO MEMORY (Crucial Step)
        # This creates the JSON files in 'memory_store/' during Run 1
        optimal_policy = result.get("optimal_policy", [])
        
        # Extract cost safely
        evaluation_str = result.get("evaluation", "")
        optimal_cost = float('inf')
        import re
        match = re.search(r'U\(w,p\*\)\s*=\s*([\d.]+)', evaluation_str)
        if match:
            optimal_cost = float(match.group(1))

        # Save to disk
        memory_manager.save_execution(
            workflow_dict=workflow_dict,
            env_dict=env_dict,
            params=params,
            optimal_policy=optimal_policy,
            evaluation_result={"best_cost": optimal_cost},
            plan=result.get("plan", ""),
            experiment_id=experiment_id
        )

        return {
            "success": True,
            "final_cost": optimal_cost,
            "policy": optimal_policy,
            "plan": result.get("plan", "")
        }

    except Exception as e:
        return {
            "success": False, 
            "error": str(e),
            "final_cost": float('inf'),
            "policy": []
        }
    

if __name__ == "__main__":
    # ========================================================================
    # INITIALIZE MEMORY SYSTEM
    # ========================================================================
    
    print("🧠 Initializing Memory System...")
    memory_manager = WorkflowMemory(memory_dir="memory_store")
    print(f"   Memory directory: {memory_manager.memory_dir}")
    
    # ========================================================================
    # LOAD DATASET FROM JSON
    # ========================================================================
    
    print("\n📂 Loading dataset from dataset/dataset.json...")
    dataset = load_dataset("dataset/dataset.json")
    print(f"   Loaded {len(dataset)} experiment configurations\n")
    
    # Limit number of experiments for testing (set to None to run all)
    threshold = 9
    
    # Iterate over all objects and evaluate each
    for idx, dataset_obj in enumerate(dataset):
        if threshold is not None and threshold <= 0:
            break
        
        try:
            calculate_experiment(dataset_obj, idx, memory_manager)
            
            if threshold is not None:
                threshold -= 1
                
        except Exception as e:
            print(f"Error while processing experiment {idx} (ID: {dataset_obj.get('id', 'unknown')}): {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("All experiments processed.")
    print(f"Memory stored in: {memory_manager.memory_dir}")
    print("="*80)