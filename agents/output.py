# agents/output.py - FULLY ALIGNED with paper specifications
import json
from agents.base_agent import BaseAgent


class OutputAgent(BaseAgent):
    def __init__(self, api_key: str, log_file: str = "agent_trace.txt"):
        super().__init__(api_key)
        self.log_file = log_file

    def format_output(self, plan: str, evaluation: str, optimal_policy, 
                     workflow_dict: dict = None, env_dict: dict = None, params: dict = None):
        policy_str = str(optimal_policy) if optimal_policy else "[]"
        
        # Build task mapping following paper notation
        task_mapping = ""
        if optimal_policy and workflow_dict:
            tasks = workflow_dict.get('tasks', {})
            locations = env_dict.get('locations', {}) if env_dict else {}
            
            mapping_lines = []
            mapping_lines.append("Optimal Placement Policy p* = {l_1, l_2, ..., l_N}:")
            mapping_lines.append("")
            
            for i, loc in enumerate(optimal_policy, start=1):
                if i in tasks:
                    loc_type = locations.get(loc, 'unknown')
                    v_i = tasks[i].get('v', 0)
                    
                    if loc == 0:
                        mapping_lines.append(f"  Task {i}: l_{i} = {loc} (IoT - Local Execution)")
                    else:
                        mapping_lines.append(f"  Task {i}: l_{i} = {loc} ({loc_type.upper()} Server)")
                    
                    mapping_lines.append(f"    v_{i} = {v_i:.2e} CPU cycles")
            
            task_mapping = "\n".join(mapping_lines)
        
        # Format environment summary with paper notation
        env_summary = self._format_env_summary(env_dict) if env_dict else "No environment details"
        params_str = self._format_params(params) if params else "No parameters"
        
        # Determine mode
        mode_desc = self._get_mode_description(params)
        
        prompt = f"""
You are the Output Agent providing final recommendations for task offloading based on the paper's framework.

## Environment Configuration (Section III-A):
{env_summary}

## Cost Model Parameters (Section III-C):
{params_str}

## Optimization Mode:
{mode_desc}

## Planner's Strategic Analysis:
{plan[:500]}...

## Evaluator's Result:
{evaluation}

## Optimal Policy Found:
{policy_str}

## Task-to-Location Mapping:
{task_mapping if task_mapping else "No valid policy found"}

## Paper Context:
The offloading cost U(w, p) is computed using Equation 8:
  U(w, p) = delta_t * T + delta_e * E

Where:
- T = CT * Delta_max (time cost via critical path, Eq. 7)
- E = CE * (ED + EV) (energy cost, Eq. 3)
  * ED = data communication energy (Eq. 4)
  * EV = task execution energy (Eq. 5)

Using Chain-of-Thought reasoning, provide a comprehensive explanation:

1. **Why is this policy optimal?**
   - How does it minimize U(w, p) according to the paper's cost model?
   - What is the balance between time (T) and energy (E) costs?
   - How does it leverage the DR, DE, VR, VE parameters?

2. **Cost Analysis**:
   - Expected time consumption (critical path through delay-DAG)
   - Expected energy consumption (data + execution)
   - Improvement over baseline (all-local execution)

3. **Placement Rationale**:
   - Which tasks are offloaded and why?
   - Which tasks remain local and why?
   - How are task dependencies (d_i,j) handled?

4. **Performance Benefits**:
   - Latency reduction from using faster processors
   - Energy savings from efficient resource allocation
   - Network overhead vs. computation savings trade-off

5. **Implementation Considerations**:
   - Critical path tasks and their placement
   - Data transfer bottlenecks
   - Robustness to environment changes
   - Monitoring and adaptation strategies

## Concise Output Requirement

Return a short, direct explanation — no chain-of-thought — using:

<summary>≤ 35-word overview</summary>
<bullets>
- key insight 1
- key insight 2
- key insight 3
</bullets>
<justification>one-line reasoning</justification>

Focus on clarity and brevity. All deep reasoning should remain internal.
   

Provide your explanation using the paper's notation and terminology.

"""
        
        # Log the prompt
        self._log_interaction("OUTPUT", prompt, None, "PROMPT")
        
        result = self.think_with_cot(prompt, return_reasoning=True)
        
        full_response = f"REASONING:\n{result['reasoning']}\n\nEXPLANATION:\n{result['answer']}"
        
        # Log the response
        self._log_interaction("OUTPUT", None, full_response, "RESPONSE")
        
        # Construct comprehensive output
        output = {
            "plan_summary": plan[:500] + ("..." if len(plan) > 500 else ""),
            "evaluation_summary": evaluation,
            "recommended_policy": list(optimal_policy) if optimal_policy else [],
            "task_mapping": task_mapping,
            "optimization_mode": mode_desc,
            "cost_model": "U(w,p) = delta_t * T + delta_e * E (Equation 8)",
            "confidence": "High" if optimal_policy else "Low",
            "reasoning": result['reasoning'],
            "explanation": result['answer']
        }
        
        return json.dumps(output, indent=2)

    def _format_env_summary(self, env_dict: dict):
        """Format environment summary with paper notation."""
        lines = []
        
        # Locations
        locations = env_dict.get('locations', {})
        if locations:
            lines.append("Locations (l):")
            for loc_id, loc_type in sorted(locations.items()):
                lines.append(f"  l={loc_id}: {loc_type.upper()}")
            lines.append("")
        
        # DR - Data Transfer Time
        dr = env_dict.get('DR', {})
        if dr:
            lines.append("DR(li, lj) - Data Time Consumption [ms/byte]:")
            sample_count = 0
            for (src, dst), rate in sorted(dr.items()):
                if src != dst and sample_count < 5:
                    lines.append(f"  DR({src},{dst}) = {rate:.6e} ms/byte")
                    sample_count += 1
            if len([1 for (s,d) in dr.keys() if s!=d]) > 5:
                lines.append(f"  ... ({len([1 for (s,d) in dr.keys() if s!=d]) - 5} more)")
            lines.append("")
        
        # DE - Data Energy
        de = env_dict.get('DE', {})
        if de:
            lines.append("DE(li) - Data Energy Consumption [mJ/byte]:")
            for loc, coeff in sorted(de.items()):
                lines.append(f"  DE({loc}) = {coeff:.6e} mJ/byte")
            lines.append("")
        
        # VR - Task Execution Time
        vr = env_dict.get('VR', {})
        if vr:
            lines.append("VR(li) - Task Time Consumption [ms/cycle]:")
            for loc, rate in sorted(vr.items()):
                lines.append(f"  VR({loc}) = {rate:.6e} ms/cycle")
            lines.append("")
        
        # VE - Task Energy
        ve = env_dict.get('VE', {})
        if ve:
            lines.append("VE(li) - Task Energy Consumption [mJ/cycle]:")
            for loc, energy in sorted(ve.items()):
                lines.append(f"  VE({loc}) = {energy:.6e} mJ/cycle")
        
        return "\n".join(lines)

    def _format_params(self, params: dict):
        """Format parameters with paper context."""
        lines = []
        lines.append("Cost Coefficients (Equations 1-2):")
        lines.append(f"  CT = {params.get('CT', 0.2)} (cost per unit time)")
        lines.append(f"  CE = {params.get('CE', 1.34)} (cost per unit energy)")
        lines.append("")
        lines.append("Mode Weights (Equation 8):")
        lines.append(f"  delta_t = {params.get('delta_t', 1)}")
        lines.append(f"  delta_e = {params.get('delta_e', 1)}")
        return "\n".join(lines)

    def _get_location_name(self, location: int, loc_type: str = None) -> str:
        """Convert location ID to human-readable name."""
        if location == 0:
            return "IoT Device (Local, l=0)"
        elif loc_type:
            return f"{loc_type.capitalize()} Server (l={location})"
        else:
            return f"Remote Server (l={location})"

    def _get_mode_description(self, params: dict) -> str:
        """Get human-readable mode description following paper."""
        if not params:
            return "Unknown mode"
        
        delta_t = params.get('delta_t', 1)
        delta_e = params.get('delta_e', 1)
        
        if delta_t == 1 and delta_e == 1:
            return "BALANCED MODE: U(w,p) = T + E (minimize both time and energy)"
        elif delta_t == 1 and delta_e == 0:
            return "LOW LATENCY MODE: U(w,p) = T (minimize time only, ignore energy)"
        elif delta_t == 0 and delta_e == 1:
            return "LOW POWER MODE: U(w,p) = E (minimize energy only, ignore time)"
        else:
            return f"CUSTOM MODE: U(w,p) = {delta_t}*T + {delta_e}*E"

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

    def run(self, state: dict):
        plan = state.get("plan", "")
        evaluation = state.get("evaluation", "")
        optimal_policy = state.get("optimal_policy", [])
        workflow_dict = state.get("workflow", {})
        env_dict = state.get("env", {})
        params = state.get("params", {})
        
        output = self.format_output(plan, evaluation, optimal_policy, 
                                    workflow_dict, env_dict, params)
        
        print("\n" + "="*60)
        print("FINAL OUTPUT (with CoT explanation):")
        print("="*60)
        print(json.dumps(json.loads(output), indent=2))
        print("="*60 + "\n")
        
        return {
            **state,
            "output": output
        }