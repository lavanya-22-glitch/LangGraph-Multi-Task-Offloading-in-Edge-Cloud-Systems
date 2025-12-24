from __future__ import annotations
from typing import Dict, List
import math

from core.workflow import Workflow
from core.environment import Environment

from core.utils import topological_sort


class UtilityEvaluator:
    """
    Compute offloading cost U(w, p) = delta_t * T + delta_e * E
    where:
      - E = CE * (ED + EV)
      - T = CT * Delta_max(delay-DAG)
    
    (Section III-C: Cost Model)
    """

    def __init__(self, CT: float = 0.2, CE: float = 1.34, delta_t: int = 1, delta_e: int = 1):
        """
        Args:
            CT: Cost per unit of execution time (default: 0.2 = 1/Tt where Tt=5ms)
            CE: Cost per unit of energy consumption (default: 1.34 = 1/Te where Te≈0.746mJ)
            delta_t: Weight for time consumption cost (0 or 1)
            delta_e: Weight for energy consumption cost (0 or 1)
        """
        self.CT = CT
        self.CE = CE
        self.delta_t = delta_t
        self.delta_e = delta_e

    # ---------------- Energy Computations (Equation 3-5) ----------------
    
    def compute_ED(self, workflow: Workflow, placement: Dict[int, int], env: Environment) -> float:
        """
        Compute ED: total data communication energy cost (Equation 4)
        
        ED = sum_{i= 1-N} [ DE(li) * (sum_{j in Ji} d_{j,i} + sum_{k in Ki} d_{i,k}) ]
        
        Args:
            workflow: Workflow object containing tasks and dependencies
            placement: Dict mapping task_id -> location_id
            env: Environment object with DE function
            
        Returns:
            Total data communication energy cost
        """
        ED = 0.0
        
        Ji = workflow.Ji()  # parents of each node
        D = workflow.D()    # all edges (i,j) -> d_{i,j}
        
        # Iterate over real tasks (1 to N)
        for i in range(1, workflow.N + 1):
            li = placement[i]
            
            # Incoming data from parents: sum_{j in Ji} d_{j,i}
            incoming = sum(D.get((j,i), 0.0) for j in Ji.get(i, []))
            
            # Outgoing data to children: sum_{k in Ki} d_{i,k}
            outgoing = sum(D.get((i, k), 0.0) for k in workflow.Ki().get(i, []))
            
            total_bytes = incoming + outgoing
            ED += env.DE(li) * total_bytes
        
        return ED

    def compute_EV(self, workflow: Workflow, placement: Dict[int, int], env: Environment) -> float:
        """
        Compute EV: total execution energy (Equation 5)
        
        EV = sum_{i = 1-N} [ vi * VE(li) ]
        
        Args:
            workflow: Workflow object containing tasks
            placement: Dict mapping task_id -> location_id
            env: Environment object with VE function
            
        Returns:
            Total execution energy cost
        """
        EV = 0.0
        V = workflow.V()  # task sizes (cc)
        
        # Iterate over real tasks (1 to N)
        for i in range(1, workflow.N + 1):
            li = placement[i]
            EV += V[i] * env.VE(li)
        
        return EV

    def compute_energy_cost(self, workflow: Workflow, placement: Dict[int, int], env: Environment) -> float:
        """
        Compute total energy consumption cost (Equation 3)
        
        E = CE * (ED + EV)
        
        Args:
            workflow: Workflow object
            placement: Dict mapping task_id -> location_id
            env: Environment object
            
        Returns:
            Total energy consumption cost
        """
        ED = self.compute_ED(workflow, placement, env)
        EV = self.compute_EV(workflow, placement, env)
        return self.CE * (ED + EV)

    # ---------------- Time Computations (Equation 6-7) ----------------
    
    def compute_delay_edge_weight(self, i: int, j: int, workflow: Workflow,
                                  placement: Dict[int, int], env: Environment) -> float:
        """
        Compute edge weight for delay-DAG (Equation 6)
        
        DΔ(i, j) = d_{i,j} * DR(li, lj) + vi * VR(li)
        
        Args:
            i: Parent task ID
            j: Child task ID
            workflow: Workflow object
            placement: Dict mapping task_id -> location_id
            env: Environment object
            
        Returns:
            Delay edge weight from task i to task j
        """
        D = workflow.D()
        V = workflow.V()
        
        d_ij = D.get((i, j), 0.0)  # data dependency from i to j
        li = placement[i]
        lj = placement[j]
        
        dr = env.DR(li, lj)
        vr = env.VR(li)
        
        if math.isinf(dr):
            return float('inf')
        
        return d_ij * dr + V[i] * vr

    def compute_critical_path_delay(self, workflow: Workflow, placement: Dict[int, int], 
                                    env: Environment) -> float:
        """
        Compute Delta_max: the maximum delay (critical path) through the delay-DAG
        
        Uses topological sort and dynamic programming to find longest path in DAG
        from entry node (0) to exit node (N+1).
        
        Args:
            workflow: Workflow object
            placement: Dict mapping task_id -> location_id
            env: Environment object
            
        Returns:
            Maximum delay (critical path length)
        """
        # Get adjacency list and topological order
        Ki = workflow.Ki()  # children of each node
        vertices = workflow.vertices()
        
        # FIXED: Pass adjacency list to topological_sort
        order = topological_sort(Ki)
        
        if order is None:
            # Graph has a cycle, return infinite cost
            return float('inf')
        
        # Initialize distances
        dist = {v: float('-inf') for v in vertices}
        dist[0] = 0.0  # Entry node
        
        # DP for longest path
        for u in order:
            if dist[u] == float('-inf'):
                continue
            
            for v in Ki.get(u, []):
                # Compute edge weight using placement
                if u == 0:
                    # Entry node: no execution cost, only data transfer
                    w_uv = 0.0 
                elif v == workflow.N + 1:
                    # Exit node: only include task u execution and data transfer
                    w_uv = self.compute_delay_edge_weight(u, v, workflow, placement, env)
                else:
                    # Regular edge
                    w_uv = self.compute_delay_edge_weight(u, v, workflow, placement, env)
                
                if math.isinf(w_uv):
                    dist[v] = float('inf')
                else:
                    dist[v] = max(dist[v], dist[u] + w_uv)
        
        return dist[workflow.N + 1]

    def compute_time_cost(self, workflow: Workflow, placement: Dict[int, int], env: Environment) -> float:
        """
        Compute total time consumption cost (Equation 7)
        
        T = CT * Delta_max(w_Δ)
        
        Args:
            workflow: Workflow object
            placement: Dict mapping task_id -> location_id
            env: Environment object
            
        Returns:
            Total time consumption cost
        """
        delta_max = self.compute_critical_path_delay(workflow, placement, env)
        if math.isinf(delta_max):
            return float('inf')
        return self.CT * delta_max

    # ---------------- Total Offloading Cost (Equation 8) ----------------
    
    def total_offloading_cost(self, workflow: Workflow, placement: Dict[int, int], 
                            env: Environment) -> float:
        """
        Compute total offloading cost (Equation 8)
        
        U(w, p) = delta_t * T + delta_e * E
        
        where:
        - T = CT * Delta_max(delay-DAG)
        - E = CE * (ED + EV)
        
        Args:
            workflow: Workflow object
            placement: Dict mapping task_id -> location_id (1 to N)
            env: Environment object
            
        Returns:
            Total offloading cost
        """
        # Add entry (0) and exit (N+1) to placement (both at IoT device)
        full_placement = {0: 0, workflow.N + 1: 0}
        full_placement.update(placement)
        
        # Compute energy and time costs
        energy = self.compute_energy_cost(workflow, full_placement, env)
        time_cost = self.compute_time_cost(workflow, full_placement, env)
        
        # Check for infinite costs
        if math.isinf(energy) or math.isinf(time_cost):
            return float('inf')
        
        # Apply mode weights (delta_t, delta_e)
        return self.delta_t * time_cost + self.delta_e * energy
    
    def evaluate(self, workflow: Workflow, placement: Dict[int, int], 
                env: Environment) -> Dict[str, float]:
        """
        Evaluate and return detailed cost breakdown
        
        Args:
            workflow: Workflow object
            placement: Dict mapping task_id -> location_id
            env: Environment object
            
        Returns:
            Dict with 'total', 'energy', 'time', 'ED', 'EV' costs
        """
        # Add entry and exit to placement
        full_placement = {0: 0, workflow.N + 1: 0}
        full_placement.update(placement)
        
        ED = self.compute_ED(workflow, full_placement, env)
        EV = self.compute_EV(workflow, full_placement, env)
        energy = self.CE * (ED + EV)
        
        delta_max = self.compute_critical_path_delay(workflow, full_placement, env)
        time_cost = self.CT * delta_max
        
        total = self.delta_t * time_cost + self.delta_e * energy
        
        return {
            'total': total,
            'energy': energy,
            'time': time_cost,
            'ED': ED,
            'EV': EV,
            'delta_max': delta_max
        }