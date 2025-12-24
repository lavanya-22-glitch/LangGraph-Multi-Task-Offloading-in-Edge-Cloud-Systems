import streamlit as st
import json
from agents.main import run_workflow
from core.network import Network, Node
from core.environment import Environment
from core.workflow import Workflow, Task

st.set_page_config(page_title="Agentic Workflow Chat", layout="wide")
st.title("Agentic Workflow Chat Interface")

# --- Sidebar setup ---
st.sidebar.header("Configuration")
query = st.sidebar.text_input("Task Description", "Find optimal offloading policy")

if st.sidebar.button("Run Agents"):
    st.session_state["run_agents"] = True

# --- Main chat container ---
chat_container = st.container()

if "run_agents" in st.session_state and st.session_state["run_agents"]:
    st.session_state["run_agents"] = False

    with chat_container:
        st.chat_message("system").markdown("### Initializing Environment...")

        # Step 1: Build environment
        network = Network()
        network.add_node(Node(0, 'edge', compute_power=10e9, energy_coeff=0.5))
        network.add_node(Node(1, 'cloud', compute_power=50e9, energy_coeff=0.2))
        network.add_link(0, 1, bandwidth=10e6, delay=0.01)
        network.add_link(1, 0, bandwidth=10e6, delay=0.01)
        network.add_link(0, 0, bandwidth=10e6, delay=0.0)
        network.add_link(1, 1, bandwidth=10e6, delay=0.0)

        env = Environment(network)
        env.randomize(seed=42)

        # Step 2: Define workflow
        tasks = [
            Task(0, size=5.0, dependencies={}),
            Task(1, size=10.0, dependencies={0: 2.0}),
            Task(2, size=8.0, dependencies={1: 1.0})
        ]
        wf = Workflow(tasks)

        # Step 3: Run agentic workflow
        st.chat_message("user").markdown(f"**User:** {query}")
        print(query, {
                "env": env.get_all_parameters(),
                "workflow": wf.to_dict(),
                "params": {"CT": 0.2, "CE": 1.34, "delta_t": 1, "delta_e": 1}
            })
        with st.chat_message("assistant"):
            st.markdown("#### Planner Agent is thinking...")
            
            result = run_workflow(query, {
                "env": env.get_all_parameters(),
                "workflow": wf.to_dict(),
                "params": {"CT": 0.2, "CE": 1.34, "delta_t": 1, "delta_e": 1}
            })

        # --- Extract intermediate results ---
        plan = result.get("plan", "")
        evaluation = result.get("evaluation", "")
        output = json.loads(result.get("output", "{}"))
        reasoning = output.get("reasoning", "")
        explanation = output.get("explanation", "")

        # --- Display step-by-step messages ---
        st.chat_message("assistant").markdown("### **Planner Agent**")
        st.markdown(f"**Plan Summary:**\n\n```\n{plan[:800]}\n```")

        st.chat_message("assistant").markdown("### **Evaluator Agent**")
        st.markdown(f"**Evaluation Summary:**\n\n```\n{evaluation}\n```")

        st.chat_message("assistant").markdown("### **Output Agent**")
        st.markdown("#### Reasoning (Chain of Thought)")
        st.info(reasoning)

        st.markdown("#### Final Explanation")
        st.success(explanation)

        # --- Final JSON output ---
        st.divider()
        st.markdown("### Full Structured Output")
        st.json(output)
