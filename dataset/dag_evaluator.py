import json
import math
import itertools
from typing import Dict, List, Tuple
from collections import defaultdict, deque

# ==================== WORKFLOW CLASS ====================
class Workflow:
    """Represents a DAG workflow with tasks and dependencies"""
    
    def __init__(self, workflow_data: dict):
        # Tasks are indexed from 1 to N (NOT 0 to N-1)
        self.tasks = {int(k): v for k, v in workflow_data['tasks'].items()}
        self.edges = workflow_data['edges']
        self.N = workflow_data['N']
        
        # Build adjacency for validation
        self._build_adjacency()
    
    def _build_adjacency(self):
        """Build complete adjacency including entry and exit nodes"""
        self._children = defaultdict(list)
        self._parents = defaultdict(list)
        
        # Add edges from data
        for edge in self.edges:
            u, v = edge['u'], edge['v']
            self._children[u].append(v)
            self._parents[v].append(u)
        
        # Add entry node (0) connections to tasks with no parents
        for task_id in range(1, self.N + 1):
            if task_id not in self._parents or len(self._parents[task_id]) == 0:
                self._children[0].append(task_id)
                self._parents[task_id].append(0)
        
        # Add exit node (N+1) connections from tasks with no children
        for task_id in range(1, self.N + 1):
            if task_id not in self._children or len(self._children[task_id]) == 0:
                self._children[task_id].append(self.N + 1)
                self._parents[self.N + 1].append(task_id)
    
    def V(self) -> Dict[int, float]:
        """Return task sizes (CPU cycles) - tasks 1 to N"""
        result = {task_id: task_data['v'] for task_id, task_data in self.tasks.items()}
        # Add entry and exit nodes with minimal size
        result[0] = 1.0
        result[self.N + 1] = 1.0
        return result
    
    def D(self) -> Dict[Tuple[int, int], float]:
        """Return data dependencies (bytes)"""
        return {(edge['u'], edge['v']): edge['bytes'] for edge in self.edges}
    
    def Ji(self) -> Dict[int, List[int]]:
        """Return parent tasks for each task"""
        return dict(self._parents)
    
    def Ki(self) -> Dict[int, List[int]]:
        """Return child tasks for each task"""
        return dict(self._children)
    
    def vertices(self) -> List[int]:
        """Return all vertices including entry (0) and exit (N+1)"""
        return list(range(0, self.N + 2))


# ==================== ENVIRONMENT CLASS ====================
class Environment:
    """Represents the edge-cloud environment parameters"""
    
    def __init__(self, env_data: dict, location_types: dict):
        self.dr_data = env_data['DR']
        self.de_data = env_data['DE']
        self.vr_data = env_data['VR']
        self.ve_data = env_data['VE']
        # Parse location types - keys should be integers
        self.location_types = {int(k): v for k, v in location_types.items()}
        self.location_ids = sorted(self.location_types.keys())
        self.num_locations = len(self.location_types)
    
    def DR(self, li: int, lj: int) -> float:
        """Data time consumption rate between locations"""
        key = f"{li},{lj}"
        return self.dr_data.get(key, 0.0 if li == lj else 0.0)
    
    def DE(self, li: int) -> float:
        """Data energy consumption rate at location"""
        return self.de_data.get(str(li), 0.0)
    
    def VR(self, li: int) -> float:
        """Task time consumption rate at location"""
        return self.vr_data.get(str(li), 0.0)
    
    def VE(self, li: int) -> float:
        """Task energy consumption rate at location"""
        return self.ve_data.get(str(li), 0.0)


# ==================== UTILITY EVALUATOR ====================
class UtilityEvaluator:
    """Compute offloading cost U(w, p) = delta_t * T + delta_e * E"""
    
    def __init__(self, CT: float = 0.2, CE: float = 1.2, delta_t: int = 1, delta_e: int = 1):
        self.CT = CT
        self.CE = CE
        self.delta_t = delta_t
        self.delta_e = delta_e
    
    def compute_ED(self, workflow: Workflow, placement: Dict[int, int], env: Environment) -> float:
        """Compute total data communication energy cost (Equation 4)"""
        ED = 0.0
        Ji = workflow.Ji()
        Ki = workflow.Ki()
        D = workflow.D()
        
        for i in range(1, workflow.N + 1):
            li = placement.get(i, 0)
            
            # Incoming data from parents: sum_{j in Ji} d_{j,i}
            incoming = sum(D.get((j, i), 0.0) for j in Ji.get(i, []))
            
            # Outgoing data to children: sum_{k in Ki} d_{i,k}
            outgoing = sum(D.get((i, k), 0.0) for k in Ki.get(i, []))
            
            total_bytes = incoming + outgoing
            ED += env.DE(li) * total_bytes
        
        return ED
    
    def compute_EV(self, workflow: Workflow, placement: Dict[int, int], env: Environment) -> float:
        """Compute total execution energy (Equation 5)"""
        EV = 0.0
        V = workflow.V()
        
        for i in range(1, workflow.N + 1):
            li = placement.get(i, 0)
            EV += V[i] * env.VE(li)
        
        return EV
    
    def compute_energy_cost(self, workflow: Workflow, placement: Dict[int, int], env: Environment) -> float:
        """Compute E = CE * (ED + EV) (Equation 3)"""
        ED = self.compute_ED(workflow, placement, env)
        EV = self.compute_EV(workflow, placement, env)
        return self.CE * (ED + EV)
    
    def topological_sort(self, adj_list: Dict[int, List[int]]) -> List[int]:
        """Topological sort using Kahn's algorithm"""
        # Get all vertices
        vertices = set(adj_list.keys())
        for children in adj_list.values():
            vertices.update(children)
        
        # Calculate in-degrees
        in_degree = {v: 0 for v in vertices}
        for u in adj_list:
            for v in adj_list[u]:
                in_degree[v] += 1
        
        # Start with vertices of in-degree 0
        queue = deque([v for v in vertices if in_degree[v] == 0])
        result = []
        
        while queue:
            u = queue.popleft()
            result.append(u)
            
            for v in adj_list.get(u, []):
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        
        # Check for cycles
        if len(result) != len(vertices):
            return None
        
        return result
    
    def compute_delay_edge_weight(self, i: int, j: int, workflow: Workflow,
                                  placement: Dict[int, int], env: Environment) -> float:
        """Compute edge weight: d_ij * DR(li, lj) + vi * VR(li) (Equation 6)"""
        D = workflow.D()
        V = workflow.V()
        
        d_ij = D.get((i, j), 0.0)
        li = placement.get(i, 0)
        lj = placement.get(j, 0)
        
        dr = env.DR(li, lj)
        vr = env.VR(li)
        
        # For entry node, no execution time
        if i == 0:
            return d_ij * dr
        
        return d_ij * dr + V[i] * vr
    
    def compute_critical_path_delay(self, workflow: Workflow, placement: Dict[int, int], 
                                    env: Environment) -> float:
        """Compute Delta_max: maximum delay through delay-DAG"""
        Ki = workflow.Ki()
        
        # Topological sort
        order = self.topological_sort(Ki)
        
        if order is None:
            return float('inf')
        
        # Initialize distances for longest path (max delay)
        dist = {}
        for v in order:
            dist[v] = 0.0
        
        # DP for longest path from entry (0) to exit (N+1)
        for u in order:
            for v in Ki.get(u, []):
                edge_weight = self.compute_delay_edge_weight(u, v, workflow, placement, env)
                dist[v] = max(dist[v], dist[u] + edge_weight)
        
        return dist.get(workflow.N + 1, float('inf'))
    
    def compute_time_cost(self, workflow: Workflow, placement: Dict[int, int], env: Environment) -> float:
        """Compute T = CT * Delta_max (Equation 7)"""
        delta_max = self.compute_critical_path_delay(workflow, placement, env)
        if math.isinf(delta_max):
            return float('inf')
        return self.CT * delta_max
    
    def total_offloading_cost(self, workflow: Workflow, placement: Dict[int, int], 
                            env: Environment) -> float:
        """Compute U(w, p) = delta_t * T + delta_e * E (Equation 8)"""
        # Add entry (0) and exit (N+1) to placement (both at IoT device location 0)
        full_placement = {0: 0, workflow.N + 1: 0}
        full_placement.update(placement)
        
        energy = self.compute_energy_cost(workflow, full_placement, env)
        time_cost = self.compute_time_cost(workflow, full_placement, env)
        
        if math.isinf(energy) or math.isinf(time_cost):
            return float('inf')
        
        return self.delta_t * time_cost + self.delta_e * energy


# ==================== OPTIMAL POLICY FINDER ====================
def find_optimal_policy(workflow: Workflow, env: Environment, 
                       CT: float, CE: float, delta_t: int, delta_e: int, 
                       debug_first: bool = True) -> Tuple[List[int], float]:
    """
    Find optimal offloading policy using brute force search over actual location IDs
    
    Returns:
        (optimal_placement_list, optimal_cost)
        where optimal_placement_list[i] = location for task (i+1)
    """
    evaluator = UtilityEvaluator(CT=CT, CE=CE, delta_t=delta_t, delta_e=delta_e)
    
    # Use actual location IDs from environment
    location_ids = env.location_ids
    N = workflow.N
    
    # Debug: Test first placement
    if debug_first:
        test_placement = {i+1: 0 for i in range(N)}
        test_cost = evaluator.total_offloading_cost(workflow, test_placement, env)
        print(f"  DEBUG: All-local placement cost = {test_cost:.6f}")
        
        # Test with first non-zero location
        if len(location_ids) > 1:
            test_placement2 = {i+1: location_ids[1] for i in range(N)}
            test_cost2 = evaluator.total_offloading_cost(workflow, test_placement2, env)
            print(f"  DEBUG: All-on-location-{location_ids[1]} cost = {test_cost2:.6f}")
    
    best_cost = float('inf')
    best_placement = None
    
    total_combinations = len(location_ids) ** N
    print(f"  Evaluating {total_combinations:,} possible placements...")
    print(f"  Location IDs: {location_ids}")
    
    checked = 0
    valid_found = 0
    
    # Iterate through all possible placements using actual location IDs
    for placement_tuple in itertools.product(location_ids, repeat=N):
        # Convert to dict: task_id (1 to N) -> location_id
        placement = {i+1: placement_tuple[i] for i in range(N)}
        
        # Compute cost
        cost = evaluator.total_offloading_cost(workflow, placement, env)
        
        # Track valid solutions
        if not math.isinf(cost):
            valid_found += 1
            if valid_found == 1:
                print(f"  ✓ First valid solution found: cost = {cost:.6f}")
        
        # Update best
        if cost < best_cost:
            best_cost = cost
            best_placement = list(placement_tuple)
        
        checked += 1
        if total_combinations > 1000 and checked % max(1, total_combinations // 20) == 0:
            progress = 100 * checked / total_combinations
            print(f"    Progress: {checked:,}/{total_combinations:,} ({progress:.1f}%) - Best: {best_cost:.2f}, Valid: {valid_found}")
    
    print(f"  Total valid solutions found: {valid_found}/{total_combinations}")
    
    if best_placement is None or math.isinf(best_cost):
        print("  ⚠️  WARNING: All placements resulted in infinite cost!")
        # Return all tasks on IoT device (location 0) as fallback
        best_placement = [0] * N
        best_cost = evaluator.total_offloading_cost(
            workflow, {i+1: 0 for i in range(N)}, env
        )
    
    return best_placement, best_cost


# ==================== MAIN PROCESSING ====================
def process_dataset(json_file_path: str, output_file_path: str, start_int: int, end_ind: int):
    """Process entire dataset and find optimal policies"""
    
    # Load dataset
    with open(json_file_path, 'r') as f:
        dataset = json.load(f)
    
    if start_int is not None and end_ind is not None:
        dataset = dataset[start_int:end_ind]
    
    print(f"Loaded {len(dataset)} workflows from dataset\n")
    
    results = []
    
    for idx, data_obj in enumerate(dataset):
        print(f"\n{'='*70}")
        print(f"Processing workflow {idx + 1}/{len(dataset)}")
        print(f"ID: {data_obj['meta']['id']}")
        print(f"{'='*70}")
        
        # Extract components
        workflow = Workflow(data_obj['workflow'])
        env = Environment(data_obj['env'], data_obj['location_types'])
        
        costs = data_obj['costs']
        mode = data_obj['mode']
        
        print(f"Workflow: N={workflow.N} tasks, {len(workflow.edges)} edges")
        print(f"Environment: {env.num_locations} locations - {env.location_types}")
        print(f"Mode: delta_t={mode['delta_t']}, delta_e={mode['delta_e']}")
        
        # Find optimal policy
        try:
            optimal_placement, optimal_cost = find_optimal_policy(
                workflow, env, 
                CT=costs['CT'], 
                CE=costs['CE'],
                delta_t=mode['delta_t'],
                delta_e=mode['delta_e'],
                debug_first=True
            )
            
            print(f"\n✓ Optimal cost: {optimal_cost:.6f}")
            print(f"✓ Optimal placement: {optimal_placement}")
            
        except Exception as e:
            print(f"\n❌ Error processing workflow: {e}")
            import traceback
            traceback.print_exc()
            optimal_placement = [0] * workflow.N
            optimal_cost = float('inf')
        
        # Store result
        result = {
            'meta': data_obj['meta'],
            'optimal_placement': optimal_placement,
            'optimal_cost': optimal_cost,
            'workflow': data_obj['workflow'],
            'location_types': data_obj['location_types'],
            'env': data_obj['env'],
            'costs': data_obj['costs'],
            'mode': data_obj['mode']
        }
        with open(output_file_path, 'a') as f:
            json.dump(result, f)
            f.write('\n')
    
        print(f"\n{'='*70}")
        print(f"✓ COMPLETED! Results saved to {output_file_path}")
        print(f"{'='*70}")
    
    # Summary statistics
    valid_costs = [r['optimal_cost'] for r in results if not math.isinf(r['optimal_cost'])]
    if valid_costs:
        print(f"\nSummary Statistics:")
        print(f"  Valid solutions: {len(valid_costs)}/{len(results)}")
        print(f"  Average cost: {sum(valid_costs)/len(valid_costs):.2f}")
        print(f"  Min cost: {min(valid_costs):.2f}")
        print(f"  Max cost: {max(valid_costs):.2f}")
    else:
        print(f"\n⚠️  WARNING: No valid solutions found for any workflow!")


# ==================== RUN ====================
if __name__ == "__main__":
    # For Google Colab:
    # 1. Upload dataset.json or mount Google Drive
    # 2. Update paths below
    
    input_file = "dataset.json"
    output_file = "output_dataset.json"
    
    print("="*70)
    print("OPTIMAL POLICY FINDER - Brute Force Search")
    print("="*70)
    print("⚠️  WARNING: Computational complexity is O(L^N)")
    print("   where L = number of locations, N = number of tasks")
    print("="*70 + "\n")
    
    # Process first workflow only for debugging, or all with max_workflows=None
    process_dataset(input_file, output_file, start_int = 0, end_ind = 5)