# runner/run_graph.py
from langgraph.graph import StateGraph, END
from agents.planner import PlannerAgent
from agents.evaluator import EvaluatorAgent
from agents.output import OutputAgent

def run_experiment(api_key: str, env_context: dict):
    planner = PlannerAgent(api_key)
    evaluator = EvaluatorAgent(api_key)
    output_agent = OutputAgent(api_key)

    graph = StateGraph(dict)
    graph.add_node("planner", lambda s: {"plan": planner.create_plan(s["env"])})
    graph.add_node("evaluator", lambda s: {"evaluation": evaluator.evaluate_plan(s["plan"], s["env"])})
    graph.add_node("output", lambda s: {"output": output_agent.format_output(s["plan"], s["evaluation"])})

    graph.set_entry_point("planner")
    graph.add_edge("planner", "evaluator")
    graph.add_edge("evaluator", "output")
    graph.add_edge("output", END)

    app = graph.compile()

    result = app.invoke({"env": env_context})
    print("\n✅ Final Output:\n", result["output"])
    return result

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()
    API_KEY = os.getenv("GOOGLE_API_KEY")

    env_context = {
        "tasks": ["T1", "T2", "T3"],
        "network": {"bandwidth": "10Mbps", "latency": "20ms"},
        "resources": {"edge": "low", "cloud": "high"}
    }

    run_experiment(API_KEY, env_context)
