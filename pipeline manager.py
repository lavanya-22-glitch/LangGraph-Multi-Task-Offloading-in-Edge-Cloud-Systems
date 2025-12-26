import os
import json
import pandas as pd
import time
import numpy as np
import dotenv
from tqdm import tqdm

# Load API Key
dotenv.load_dotenv()

# Import core modules
from core.dag_generator import generate_random_workflow
from core.cost_eval import UtilityEvaluator
from agents.main import run_workflow_headless

# ==============================================================================
#  REALISTIC PHYSICS CONSTANTS (For constructing the Environment)
# ==============================================================================
VR_PROFILE = {"iot": 1.2e-7, "edge": 1.2e-8, "cloud": 1.2e-9}
VE_PROFILE = {"iot": 6.0e-7, "edge": 2.0e-7, "cloud": 1.0e-7}
DE_PROFILE = {"iot": 0.005,  "edge": 1e-5,   "cloud": 1e-5}
DR_PROFILE = {
    "local": 0.0,
    "fast": 0.0001, # IoT <-> Edge
    "slow": 0.0020  # IoT <-> Cloud
}

# ==============================================================================
#  SCENARIOS
# ==============================================================================
PIPELINE_SCENARIOS = [
    {"name": "Tiny_Graph", "n": 5, "edge_prob": 0.2},
    {"name": "Small_Graph", "n": 7, "edge_prob": 0.3},
    {"name": "Medium_Graph", "n": 10, "edge_prob": 0.3},
    {"name": "Large_Graph", "n": 12, "edge_prob": 0.3},
    {"name": "Sparse_Dependencies", "n": 8, "edge_prob": 0.1},
    {"name": "Dense_Dependencies", "n": 8, "edge_prob": 0.8},
    {"name": "Pipeline_Structure", "n": 8, "alpha": 3.0},
    {"name": "Parallel_Structure", "n": 8, "alpha": 0.3},
    {"name": "Compute_Heavy", "n": 8, "CCR": 0.5},
    {"name": "Data_Heavy", "n": 8, "CCR": 2.0},
    {"name": "High_Bandwidth", "n": 8, "B_mb_s": 50.0},
    {"name": "Low_Bandwidth", "n": 8, "B_mb_s": 1.0},
    {"name": "Powerful_Cloud", "n": 8, "Cb_mips": 50000},
    {"name": "Weak_Cloud", "n": 8, "Cb_mips": 5000},
    {"name": "No_Remote_Nodes", "n": 6, "num_remote": 0}, 
    {"name": "All_Cloud_Nodes", "n": 6, "num_remote": 5},
    {"name": "Stress_Test_Huge", "n": 20, "edge_prob": 0.4},
]

def build_experiment_object(dag_data, scenario_name):
    """
    Wraps a raw DAG into a full Experiment Object with Environment details.
    """
    # 1. Define Locations (Standard Setup: 1 IoT, 1 Cloud, 2 Edge)
    # We randomize the Edge/Cloud count slightly based on scenario name if needed
    locations = {0: "iot", 1: "cloud", 2: "edge", 3: "edge"}
    
    if "No_Remote" in scenario_name:
        locations = {0: "iot"}
    elif "All_Cloud" in scenario_name:
        locations = {0: "iot", 1: "cloud", 2: "cloud", 3: "cloud", 4: "cloud", 5: "cloud"}

    # 2. Build Environment Matrices
    # VR (Speed) & VE (Energy)
    VR = {str(k): VR_PROFILE[v] for k, v in locations.items()}
    VE = {str(k): VE_PROFILE[v] for k, v in locations.items()}
    DE = {str(k): DE_PROFILE[v] for k, v in locations.items()}
    
    # DR (Data Rate / Latency)
    DR = {}
    for u, type_u in locations.items():
        for v, type_v in locations.items():
            key = f"{u},{v}"
            if u == v:
                DR[key] = 0.0
            elif "cloud" in [type_u, type_v] and "iot" in [type_u, type_v]:
                DR[key] = DR_PROFILE["slow"] # IoT <-> Cloud is slow
            else:
                DR[key] = DR_PROFILE["fast"] # Edge/IoT local is fast

    # 3. Construct Full Object
    return {
        "workflow": dag_data, # The raw DAG goes here
        "location_types": {str(k): v for k, v in locations.items()},
        "env": {
            "VR": VR,
            "VE": VE,
            "DE": DE,
            "DR": DR
        },
        "costs": {"CT": 0.2, "CE": 1.2},
        "mode": {"delta_t": 1, "delta_e": 1},
        "meta": {"id": f"pipeline_{scenario_name}", "seed": 42}
    }

def run_data_pipeline():
    results = []
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    print(f"🚀 Starting Research Data Pipeline ({len(PIPELINE_SCENARIOS)} Scenarios)")
    print(f"📄 Logging to: pipeline_results_{timestamp}.csv\n")

    for i, scenario in enumerate(tqdm(PIPELINE_SCENARIOS, desc="Processing Pipelines")):
        
        try:
            # 1. GENERATE RAW DAG
            raw_dag = generate_random_workflow(
                n=scenario.get("n", 6),
                edge_prob=scenario.get("edge_prob", 0.25),
                alpha=scenario.get("alpha", 1.0),
                CCR=scenario.get("CCR", 0.5),
                seed=i+100
            )
            
            # 2. WRAP INTO EXPERIMENT OBJECT
            experiment_data = build_experiment_object(raw_dag, scenario['name'])
            
            # 3. PROCESS (Run Agent)
            start_time = time.time()
            agent_out = run_workflow_headless(experiment_data)
            duration = time.time() - start_time
            
            # 4. HANDLE RESULTS
            if agent_out.get('success'):
                agent_cost = agent_out['final_cost']
                policy = agent_out.get('policy', [])     # <--- CHANGE 1: Capture Policy
                status = "Success"
            else:
                agent_cost = 0.0
                policy = []                              # <--- CHANGE 1: Handle Failure
                status = "Failed"
                print(f"Error in {scenario['name']}: {agent_out.get('error')}")

            results.append({
                "Scenario": scenario['name'],
                "Tasks": scenario.get("n"),
                "Agent_Cost": round(agent_cost, 4),
                "Optimal_Policy": str(policy),           # <--- CHANGE 2: Add to Results
                "Status": status,
                "Time_sec": round(duration, 2)
            })
            
            # Save incrementally
            pd.DataFrame(results).to_csv(f"pipeline_results_{timestamp}.csv", index=False)
            
            # Sleep slightly to avoid Rate Limits
            time.sleep(2)

        except Exception as e:
            print(f"❌ Critical Pipeline Failure on {scenario['name']}: {e}")

    print(f"\n✅ Pipeline Finished.")
    # CHANGE 3: Update Print Statement to show Policy
    print(pd.DataFrame(results)[["Scenario", "Agent_Cost", "Optimal_Policy"]])

if __name__ == "__main__":
    run_data_pipeline()