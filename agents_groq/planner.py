# agents_groq/planner.py - UPDATED with memory-based few-shot prompting
from agents_groq.base_agent import BaseAgent
import json


class PlannerAgent(BaseAgent):
    """Planner agent with Chain-of-Thought reasoning and memory-based few-shot prompting."""

    def __init__(self, api_key: str, log_file: str = "agent_trace.txt", memory_manager=None):
        super().__init__(api_key)
        self.log_file = log_file
        self.memory_manager = memory_manager  # WorkflowMemory instance

    def _format_env_details(self, env: dict):
        """Format environment details following paper Section III-A."""
        details = []
        
        # Extract location types
        locations = env.get('locations', {})
        if locations:
            details.append("Available Locations (l):")
            for loc_id, loc_type in sorted(locations.items()):
                details.append(f"  l={loc_id}: {loc_type.upper()}")
            details.append("")
        
        # DR: Data Time Consumption (ms/byte) - Paper Section III-A
        dr = env.get('DR', {})
        if dr:
            details.append("DR(li, lj) - Data Time Consumption (ms/byte):")
            details.append("  Time to move 1 byte of data between locations")
            for (src, dst), rate in sorted(dr.items()):
                if src != dst:
                    details.append(f"    DR({src}, {dst}) = {rate:.6e} ms/byte")
            details.append("")
        
        # DE: Data Energy Consumption (mJ/byte) - Paper Section III-A
        de = env.get('DE', {})
        if de:
            details.append("DE(li) - Data Energy Consumption (mJ/byte):")
            details.append("  Energy for processing 1 byte at location")
            for loc, coeff in sorted(de.items()):
                details.append(f"    DE({loc}) = {coeff:.6e} mJ/byte")
            details.append("")
        
        # VR: Task Time Consumption (ms/cycle) - Paper Section III-A
        vr = env.get('VR', {})
        if vr:
            details.append("VR(li) - Task Time Consumption (ms/cycle):")
            details.append("  Time to execute 1 CPU cycle at location")
            for loc, rate in sorted(vr.items()):
                details.append(f"    VR({loc}) = {rate:.6e} ms/cycle")
            details.append("")
        
        # VE: Task Energy Consumption (mJ/cycle) - Paper Section III-A
        ve = env.get('VE', {})
        if ve:
            details.append("VE(li) - Task Energy Consumption (mJ/cycle):")
            details.append("  Energy per CPU cycle at location")
            for loc, energy in sorted(ve.items()):
                details.append(f"    VE({loc}) = {energy:.6e} mJ/cycle")
        
        return "\n".join(details)
    
    def _format_workflow_details(self, workflow: dict):
        """Format workflow details following paper Section III-B."""
        tasks = workflow.get('tasks', {})
        edges = workflow.get('edges', {})
        N = workflow.get('N', 0)
        
        details = []
        details.append("Workflow DAG: w = {V, D}")
        details.append(f"  N = {N} (number of real tasks, excluding entry v_0 and exit v_{N+1})")
        details.append("")
        
        # Format each task following paper notation
        for task_id in sorted(tasks.keys()):
            task_data = tasks[task_id]
            v_i = task_data.get('v', 0)
            
            details.append(f"Task {task_id}:")
            details.append(f"  v_{task_id} = {v_i:.2e} CPU cycles")
            
            # Find J_i (parents) and K_i (children)
            parents = [j for (j, k), _ in edges.items() if k == task_id]
            children = [k for (j, k), _ in edges.items() if j == task_id]
            
            if parents:
                details.append(f"  J_{task_id} (parents): {{{', '.join(map(str, parents))}}}")
            else:
                details.append(f"  J_{task_id} (parents): ∅")
            
            if children:
                details.append(f"  K_{task_id} (children): {{{', '.join(map(str, children))}}}")
                details.append(f"  Data dependencies d_{{i,j}}:")
                for k in children:
                    d_ij = edges.get((task_id, k), 0)
                    details.append(f"    d_{{{task_id},{k}}} = {d_ij:.2e} bytes")
            else:
                details.append(f"  K_{task_id} (children): ∅")
            
            details.append("")
        
        return "\n".join(details)

    def create_plan(self, env: dict, workflow: dict, params: dict):
        """Create a detailed plan using Chain-of-Thought reasoning with few-shot examples."""
        
        env_details = self._format_env_details(env)
        workflow_details = self._format_workflow_details(workflow)
        
        # Format parameters with paper context
        params_formatted = [
            "Cost Coefficients (Equations 1-2):",
            f"  CT = {params.get('CT', 0.2)} (cost per unit time)",
            f"  CE = {params.get('CE', 1.34)} (cost per unit energy)",
            "",
            "Mode Weights (Equation 8):",
            f"  delta_t = {params.get('delta_t', 1)} (time weight)",
            f"  delta_e = {params.get('delta_e', 1)} (energy weight)",
            ""
        ]
        
        # Add mode description
        delta_t = params.get('delta_t', 1)
        delta_e = params.get('delta_e', 1)
        if delta_t == 1 and delta_e == 1:
            params_formatted.append("  Mode: BALANCED (minimize both time and energy)")
        elif delta_t == 1 and delta_e == 0:
            params_formatted.append("  Mode: LOW LATENCY (minimize time only)")
        elif delta_t == 0 and delta_e == 1:
            params_formatted.append("  Mode: LOW POWER (minimize energy only)")
        
        params_str = "\n".join(params_formatted)
        
        # Retrieve similar executions for few-shot prompting
        few_shot_examples = ""
        learning_prompt = ""
        
        if self.memory_manager:
            similar_executions = self.memory_manager.retrieve_similar_executions(
                workflow, env, params, top_k=3
            )
            
            if similar_executions:
                few_shot_examples = self.memory_manager.format_few_shot_examples(similar_executions)
                learning_prompt = """
**Learning from Similar Cases:**
Based on the historical examples above, identify patterns that might apply to the current scenario:
- What placement strategies worked well in similar workflows?
- How did the optimal policies balance local execution vs. offloading?
- What trade-offs were made between time and energy costs?

Apply these insights to guide your analysis of the current scenario.
"""
            else:
                few_shot_examples = "## Historical Similar Cases:\nNo similar historical cases found. Analyzing from first principles.\n"
        
        prompt = f"""
You are the Planner Agent in a multi-agent system for task offloading optimization.

Your job is to analyze the task offloading problem and create a comprehensive plan using Chain-of-Thought reasoning.

{few_shot_examples}

## Current Scenario:

### Environment Configuration (Section III-A of the paper):
{env_details}

### Workflow Structure (Section III-B - DAG-based Application Model):
{workflow_details}

### Cost Model Parameters (Section III-C):
{params_str}

## Your Task:
Analyze this edge-cloud offloading scenario step-by-step following the paper's framework:

1. **Environment Analysis**: 
   - Identify DR (Data Time Consumption - ms/byte) characteristics
   - Assess DE (Data Energy Consumption - mJ/byte) at each location
   - Evaluate VR (Task Time Consumption - ms/cycle) capabilities
   - Review VE (Task Energy Consumption - mJ/cycle) profiles
   - Count available edge servers (E) and cloud servers (C)

2. **Workflow DAG Analysis**:
   - Number of real tasks (N) excluding entry/exit nodes
   - Task sizes (v_i in CPU cycles)
   - Data dependencies (d_i,j in bytes)
   - Critical path identification
   - Parent set J_i and children set K_i for each task

3. **Cost Components (Equations 3-8)**:
   - Energy Cost: E = CE * (ED + EV)
     * ED from data communication (Eq. 4)
     * EV from task execution (Eq. 5)
   - Time Cost: T = CT * Delta_max (Eq. 7)
     * Critical path through delay-DAG (Eq. 6)
   - Total: U(w,p) = delta_t * T + delta_e * E (Eq. 8)

4. **Mode-Specific Strategy**:
   - Low Latency Mode (delta_t=1, delta_e=0): Minimize execution time
   - Low Power Mode (delta_t=0, delta_e=1): Minimize energy consumption  
   - Balanced Mode (delta_t=1, delta_e=1): Optimize both objectives

5. **Placement Strategy Recommendations**:
   - Which tasks should remain local (l_i=0)?
   - Which tasks benefit from edge offloading?
   - Which tasks justify cloud offloading despite higher latency?
   - Should dependent tasks be co-located to reduce data transfer?

{learning_prompt}

Provide a structured, detailed plan that will guide the evaluator agent in finding the optimal placement policy p = [l_1, l_2, ..., l_N].

## Concise Output Requirement (Do NOT change any internal analysis)

After completing the full Chain-of-Thought reasoning internally:

RETURN ONLY:
- A short final plan summary (≤ 40 words)
- 3–6 direct, action-oriented placement strategy bullets (≤ 12 words each)
- No detailed chain-of-thought in the output

Format:
<summary>...</summary>
<bullets>
- ...
- ...
</bullets>

Think step-by-step internally but do NOT reveal the reasoning.
"""
        
        # Log the prompt
        self._log_interaction("PLANNER", prompt, None, "PROMPT")
        
        # Use CoT reasoning
        result = self.think_with_cot(prompt, return_reasoning=True)
        if isinstance(result, dict):
            reasoning = result.get('reasoning', 'No reasoning provided')
            answer = result.get('answer', 'No answer provided')
        else:
            reasoning = str(result)
            answer = str(result)
        
        full_response = f"""
## Chain-of-Thought Reasoning:
{reasoning}

## Strategic Plan for Evaluator:
{answer}
"""
        
        # Log the response
        self._log_interaction("PLANNER", None, full_response, "RESPONSE")
        
        return full_response

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
        """
        The PlannerAgent generates a plan based on the environment,
        using Chain-of-Thought reasoning and memory-based few-shot learning.
        """
        env = state.get("env", {})
        workflow = state.get("workflow", {})
        params = state.get("params", {})
        
        plan = self.create_plan(env, workflow, params)

        new_state = dict(state)
        new_state["plan"] = plan
        
        print("\n" + "="*60)
        print("PLANNER OUTPUT (with CoT reasoning + Few-Shot Learning):")
        print("="*60)
        print(plan[:500] + "..." if len(plan) > 500 else plan)
        print("="*60 + "\n")
        
        return new_state