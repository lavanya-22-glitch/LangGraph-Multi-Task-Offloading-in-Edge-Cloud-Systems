# agents/evaluator.py - FIXED: String key handling for constraints
import itertools
import math
from core.workflow import Workflow
from core.environment import Environment
from core.cost_eval import UtilityEvaluator
from agents_groq.base_agent import BaseAgent
import json


class EvaluatorAgent(BaseAgent):
    """Evaluator agent with CoT-guided heuristic search using UtilityEvaluator."""

    def __init__(self, api_key: str, log_file: str = "agent_trace.txt"):
        super().__init__(api_key)
        self.log_file = log_file

    def _create_environment(self, env_dict: dict) -> Environment:
        """Create Environment object from dictionary."""
        locations_types = env_dict.get('locations', {})
        DR_map = env_dict.get('DR', {})
        DE_map = env_dict.get('DE', {})
        VR_map = env_dict.get('VR', {})
        VE_map = env_dict.get('VE', {})
        
        return Environment.from_matrices(
            types=locations_types,
            DR_matrix=DR_map,
            DE_vector=DE_map,
            VR_vector=VR_map,
            VE_vector=VE_map
        )

    def _format_env_for_prompt(self, env_dict: dict):
        """Format environment details comprehensively for LLM."""
        details = []
        
        # Locations
        locations = env_dict.get('locations', {})
        if locations:
            details.append("Available Locations:")
            for loc_id, loc_type in sorted(locations.items()):
                details.append(f"  Location {loc_id}: {loc_type.upper()} (l_{loc_id})")
                if loc_id == 0:
                    details.append(f"    → IoT device (local execution, no offloading)")
                else:
                    details.append(f"    → Remote server (offloading target)")
            details.append("")
        
        # DR - Data Transfer Time
        dr = env_dict.get('DR', {})
        if dr:
            details.append("Data Transfer Characteristics DR(li, lj) [ms/byte]:")
            for (src, dst), rate in sorted(dr.items()):
                if src != dst:
                    latency_per_mb = rate * 1e6  # Convert to ms/MB
                    details.append(f"  {src}→{dst}: {rate:.6e} ms/byte ({latency_per_mb:.3f} ms/MB)")
            details.append("")
        
        # DE - Data Energy
        de = env_dict.get('DE', {})
        if de:
            details.append("Data Energy Consumption DE(li) [mJ/byte]:")
            for loc, coeff in sorted(de.items()):
                details.append(f"  Location {loc}: {coeff:.6e} mJ/byte")
            details.append("")
        
        # VR - Computation Speed
        vr = env_dict.get('VR', {})
        if vr:
            details.append("Task Execution Speed VR(li) [ms/cycle]:")
            for loc, rate in sorted(vr.items()):
                ghz = 1.0 / (rate * 1e6) if rate > 0 else float('inf')
                details.append(f"  Location {loc}: {rate:.6e} ms/cycle (≈{ghz:.1f} GHz)")
            details.append("")
        
        # VE - Task Energy
        ve = env_dict.get('VE', {})
        if ve:
            details.append("Task Energy Consumption VE(li) [mJ/cycle]:")
            for loc, energy in sorted(ve.items()):
                details.append(f"  Location {loc}: {energy:.6e} mJ/cycle")
        
        return "\n".join(details)

    def get_llm_guided_heuristics(self, workflow_dict: dict, env_dict: dict, plan: str, params: dict):
        """Use LLM with CoT to generate heuristic guidance for policy search."""
        tasks = workflow_dict.get('tasks', {})
        edges = workflow_dict.get('edges', {})
        N = workflow_dict.get('N', 0)
        
        # FIX(1): use actual location IDs (not 0..n-1)
        locations = env_dict.get('locations', {})
        location_ids = sorted(locations.keys())

        env_details = self._format_env_for_prompt(env_dict)
        
        task_details = []
        for task_id in sorted(tasks.keys()):
            task_data = tasks[task_id]
            v_i = task_data.get('v', 0)
            parents = [j for (j, k), _ in edges.items() if k == task_id]
            children_deps = [(k, edges[(task_id, k)]) for (j, k), _ in edges.items() if j == task_id]
            
            task_details.append(f"\nTask {task_id}:")
            task_details.append(f"  v_{task_id} = {v_i:.2e} CPU cycles")
            if parents:
                task_details.append(f"  Depends on: Tasks {parents}")
            if children_deps:
                task_details.append(f"  Data output to:")
                for k, d_ik in children_deps:
                    task_details.append(f"    Task {k}: d_{{{task_id},{k}}} = {d_ik:.2e} bytes")
        
        prompt = f"""
You are helping optimize task offloading decisions for an edge-cloud system following the paper's framework.

## Environment Configuration:
{env_details}

## Workflow DAG (N = {N} tasks):
{chr(10).join(task_details)}

## Optimization Parameters:
{json.dumps(params, indent=2)}

## Planner's Strategic Analysis:
{plan[:800]}

## Your Task:
Generate 3-5 intelligent candidate placement policies p = {{l_1, l_2, ..., l_{N}}} using ONLY these location IDs: {location_ids}

Provide candidate policies as lists: [l_1, l_2, ..., l_{N}]

## Concise Output Requirement

Return only:
- A <summary> (≤ 25 words) giving the main insight
- A <policies> section listing 3–5 candidate policies

Format:
<summary>...</summary>
<policies>
[p1, p2, ..., pN]
[p1, p2, ..., pN]
</policies>

Do NOT output chain-of-thought. Think internally only.
"""
        self._log_interaction("EVALUATOR", prompt, None, "PROMPT")
        result = self.think_with_cot(prompt, return_reasoning=True)
        full_response = f"REASONING:\n{result['reasoning']}\n\nCANDIDATE POLICIES:\n{result['answer']}"
        self._log_interaction("EVALUATOR", None, full_response, "RESPONSE")
        
        print("\n" + "="*60)
        print("LLM HEURISTIC REASONING:")
        print("="*60)
        print(result['reasoning'][:400] + "...")
        print("="*60 + "\n")

        # FIX(1): validate against actual IDs
        policies = self._parse_policies_from_text(result['answer'], N, location_ids)
        return policies

    # FIX(1): change validator to use valid_location_ids
    def _parse_policies_from_text(self, text: str, n_tasks: int, valid_location_ids):
        """Extract policy suggestions from LLM text output."""
        import re
        valid = set(valid_location_ids)
        policies = []

        pattern = r'[\[\(](\d+(?:\s*,\s*\d+)*)[\]\)]'
        matches = re.findall(pattern, text or "")
        for match in matches:
            try:
                policy = [int(x.strip()) for x in match.split(',')]
                if len(policy) == n_tasks and all(loc in valid for loc in policy):
                    policies.append(tuple(policy))
            except:
                continue

        seen = set()
        unique_policies = []
        for p in policies:
            if p not in seen:
                seen.add(p)
                unique_policies.append(p)
        return unique_policies[:5]

    def _log_interaction(self, agent: str, prompt: str, response: str, msg_type: str):
        """Log agent interactions to file."""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            if msg_type == "PROMPT":
                f.write("\n" + "="*80 + "\n")
                f.write(f"{agent} AGENT - PROMPT\n")
                f.write("="*80 + "\n")
                f.write(prompt)
                f.write("\n" + "="*80 + "\n\n")
            elif msg_type == "RESPONSE":
                f.write("\n" + "="*80 + "\n")
                f.write(f"{agent} AGENT - RESPONSE\n")
                f.write("="*80 + "\n")
                f.write(response)
                f.write("\n" + "="*80 + "\n\n")

    # FIX(2): constraints helper with proper key type handling
    @staticmethod
    def _violates_constraints(policy_tuple, fixed, allowed):
        """
        Check if policy violates constraints.
        
        Args:
            policy_tuple: tuple of location assignments (l_1, l_2, ..., l_N)
            fixed: {task_id -> required_location_id} where task_id can be int or str
            allowed: {task_id -> iterable-of-allowed-location-ids} where task_id can be int or str
        
        Returns:
            True if policy violates constraints, False otherwise
        """
        N = len(policy_tuple)
        
        # Normalize fixed constraints (convert string keys to int)
        if fixed:
            normalized_fixed = {}
            for t, loc in fixed.items():
                try:
                    task_id = int(t) if isinstance(t, str) else t
                    normalized_fixed[task_id] = int(loc)
                except (ValueError, TypeError):
                    print(f"⚠️  Warning: Invalid fixed constraint: {t} -> {loc}")
                    continue
            
            # Check fixed constraints
            for task_id, required_loc in normalized_fixed.items():
                if 1 <= task_id <= N:
                    actual_loc = policy_tuple[task_id - 1]
                    if actual_loc != required_loc:
                        return True
        
        # Normalize allowed constraints (convert string keys to int)
        if allowed:
            normalized_allowed = {}
            for t, allowed_locs in allowed.items():
                try:
                    task_id = int(t) if isinstance(t, str) else t
                    normalized_allowed[task_id] = set(int(loc) for loc in allowed_locs)
                except (ValueError, TypeError):
                    print(f"⚠️  Warning: Invalid allowed constraint: {t} -> {allowed_locs}")
                    continue
            
            # Check allowed constraints
            for task_id, allowed_set in normalized_allowed.items():
                if 1 <= task_id <= N:
                    actual_loc = policy_tuple[task_id - 1]
                    if actual_loc not in allowed_set:
                        return True
        
        return False

    def find_best_policy(self, workflow_dict: dict, env_dict: dict, params: dict, plan: str = ""):
        """
        Search for optimal placement using CoT-guided heuristics + exhaustive search.
        Uses UtilityEvaluator to compute offloading costs following the paper's equations.
        """
        if workflow_dict is None:
            raise ValueError("workflow_dict is None")

        workflow = Workflow.from_experiment_dict(workflow_dict)
        env = self._create_environment(env_dict)
        
        N = workflow.N

        # FIX(1): derive actual location IDs once
        location_ids = sorted((env_dict.get('locations') or {}).keys())
        n_locations = len(location_ids)

        # Evaluator params
        CT = params.get('CT', 0.2)
        CE = params.get('CE', 1.34)
        delta_t = params.get('delta_t', 1)
        delta_e = params.get('delta_e', 1)
        evaluator = UtilityEvaluator(CT=CT, CE=CE, delta_t=delta_t, delta_e=delta_e)

        print(f"\n{'='*60}")
        print(f"EVALUATOR: Searching for optimal offloading policy")
        print(f"  Tasks (N): {N}")
        print(f"  Locations: {n_locations} with IDs {location_ids}")
        print(f"  Cost Model: U(w,p) = {delta_t}*T + {delta_e}*E")
        print(f"  CT={CT}, CE={CE}")
        
        # Display constraints if any
        fixed = params.get("fixed_locations", {})
        allowed = params.get("allowed_locations", None)
        if fixed:
            print(f"  Fixed Constraints: {fixed}")
        if allowed:
            print(f"  Allowed Constraints: {allowed}")
        
        print(f"{'='*60}\n")

        # LLM-guided candidates
        print("Using Chain-of-Thought to generate intelligent candidate policies...")
        llm_candidates = self.get_llm_guided_heuristics(workflow_dict, env_dict, plan, params)

        # Systematic candidates (FIX(1): use real IDs)
        systematic_candidates = []
        for lid in location_ids:
            systematic_candidates.append(tuple(lid for _ in range(N)))  # all on each available location

        # simple round-robin seeds over actual IDs
        for start in range(min(n_locations, 3)):
            cand = tuple(location_ids[(start + i) % n_locations] for i in range(N))
            systematic_candidates.append(cand)

        # Combine
        candidates = llm_candidates + [c for c in systematic_candidates if c not in llm_candidates]

        # Exhaustive completion (FIX(1): over real IDs)
        max_exhaustive = 10000
        total_combos = n_locations ** N
        if total_combos <= max_exhaustive:
            print(f"✓ Problem size allows exhaustive search ({total_combos} combinations)")
            all_candidates = list(itertools.product(location_ids, repeat=N))
            candidates = list(set(candidates + all_candidates))
        else:
            print(f"⚠  Problem too large for exhaustive search ({total_combos} combinations)")
            print(f"  Using {len(candidates)} LLM-guided + heuristic candidates")

        # FIX(2): read constraints
        fixed = params.get("fixed_locations", {}) or {}
        allowed = params.get("allowed_locations", None)

        print(f"\nEvaluating {len(candidates)} candidate policies using UtilityEvaluator...")
        print(f"  Computing U(w,p) via Equations 3-8 from paper\n")
        
        best_policy = None
        best_cost = float("inf")
        evaluated = 0
        skipped = 0

        for placement_tuple in candidates:
            # FIX(2): enforce constraints
            if self._violates_constraints(placement_tuple, fixed, allowed):
                skipped += 1
                continue
            try:
                placement_dict = {i: placement_tuple[i-1] for i in range(1, N + 1)}
                cost = evaluator.total_offloading_cost(workflow, placement_dict, env)
                evaluated += 1
                if cost is None or (isinstance(cost, float) and math.isinf(cost)):
                    skipped += 1
                    continue
                if cost < best_cost:
                    best_cost = cost
                    best_policy = placement_tuple
                    print(f"  ✓ New best: {best_policy} with U(w,p) = {best_cost:.6f}")
            except Exception as e:
                skipped += 1
                print(f"  ⚠️  Error evaluating {placement_tuple}: {e}")
                continue

        return {
            "best_policy": best_policy,
            "best_cost": best_cost,
            "evaluated": evaluated,
            "skipped": skipped
        }

    def run(self, state: dict):
        workflow_dict = state.get("workflow")
        env_dict = state.get("env", {})
        params = state.get("params", {})
        plan = state.get("plan", "")

        if not isinstance(workflow_dict, dict):
            raise ValueError("workflow_dict must be a dictionary")

        result = self.find_best_policy(workflow_dict, env_dict, params, plan)

        if result["best_policy"] is None:
            evaluation = f"No finite-cost policy found. Evaluated={result['evaluated']}, Skipped={result['skipped']}"
            optimal_policy = []
        else:
            evaluation = f"Optimal policy found: U(w,p*) = {result['best_cost']:.6f}"
            optimal_policy = list(result['best_policy'])

        print(f"\n{'='*60}")
        print(f"EVALUATION COMPLETE:")
        print(f"  {evaluation}")
        print(f"  Evaluated: {result['evaluated']} policies")
        print(f"  Skipped: {result['skipped']} policies")
        print(f"{'='*60}\n")

        return {
            **state,
            "evaluation": evaluation,
            "optimal_policy": optimal_policy
        }