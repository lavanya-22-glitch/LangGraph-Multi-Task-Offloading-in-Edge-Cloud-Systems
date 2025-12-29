import inspect
import warnings
from typing import Any, Dict, List, Union, Tuple
import numpy as np

# --- EDGE COMPUTING CORE IMPORTS ---
from core.workflow import Workflow
from agents.planner import PlannerAgent
from agents.evaluator import EvaluatorAgent

class QuestionAnsweringArgumentHandler:
    """
    Handles arguments for the QuestionAnsweringPipeline.
    Normalizes inputs to the expected {'question': ..., 'context': ...} format.
    """
    def __call__(self, *args, **kwargs):
        if "question" in kwargs and "context" in kwargs:
            return [{"question": kwargs["question"], "context": kwargs["context"]}]
        elif len(args) == 1 and isinstance(args[0], dict):
            return [args[0]]
        return args[0] if args else []

class QuestionAnsweringPipeline:
    """
    Question Answering pipeline using `PlannerAgent`. 
    Extracts the optimal configuration (Answer) given a set of constraints (Context).

    Example:
    ```python
    >>> from pipelines import pipeline
    >>> oracle = pipeline(task="policy-extraction")
    >>> oracle(question="Optimal policy?", context="High latency on Edge...")
    {'score': 0.91, 'answer': '[0, 0, 0, 0]'}
    ```
    """

    default_input_names = "question,context"
    handle_impossible_answer = False

    def __init__(self, api_key: str, **kwargs):
        self.planner = PlannerAgent(api_key=api_key)
        self.evaluator = EvaluatorAgent(api_key=api_key)
        self._args_parser = QuestionAnsweringArgumentHandler()

    def _sanitize_parameters(self, top_k=None, handle_impossible_answer=None, **kwargs):
        postprocess_params = {}
        if top_k is not None:
             postprocess_params["top_k"] = top_k
        if handle_impossible_answer is not None:
            postprocess_params["handle_impossible_answer"] = handle_impossible_answer
        return {}, {}, postprocess_params

    def __call__(self, *args, **kwargs):
        """
        Answer the question(s) given as inputs by using the context(s).
        """
        examples = self._args_parser(*args, **kwargs)
        if isinstance(examples, (list, tuple)) and len(examples) == 1:
             examples = examples[0]

        preprocess_params, _, postprocess_params = self._sanitize_parameters(**kwargs)
        
        # 1. Preprocess
        model_inputs = self.preprocess(examples)
        
        # 2. Forward
        model_outputs = self._forward(model_inputs)
        
        # 3. Postprocess
        return self.postprocess(model_outputs, **postprocess_params)

    def preprocess(self, inputs):
        """
        Convert Graph Constraints into a 'Context' string.
        """
        # Unpack strict QA format into our workflow format
        # In this architecture, we expect 'context' to hold the env/workflow dict
        context_data = inputs.get("context")
        if isinstance(context_data, str):
             # Handle string case if passed directly
             pass 
             
        workflow = context_data.get("workflow") if isinstance(context_data, dict) else inputs.get("workflow")
        env = context_data.get("env") if isinstance(context_data, dict) else inputs.get("env")
        
        return {
            "example": inputs,
            "workflow": workflow,
            "env": env
        }

    def _forward(self, inputs):
        """
        The Planner acts as the Reader model.
        """
        plan = self.planner.create_plan(
            inputs["workflow"], inputs["env"], []
        )
        return {
            "start_logits": plan, 
            "workflow": inputs["workflow"],
            "env": inputs["env"]
        }

    def postprocess(self, model_outputs, top_k=1, handle_impossible_answer=False):
        """
        Extract the exact policy vector from the fuzzy LLM plan.
        """
        plan = model_outputs["start_logits"]
        
        result = self.evaluator.find_best_policy(
             model_outputs["workflow"],
             model_outputs["env"],
             params={},
             plan=plan
        )
        
        best_policy = result["best_policy"]
        best_cost = result["best_cost"]
        
        # Format as QA answer
        return {
            "score": 1.0 / (best_cost + 1e-5),
            "start": 0,
            "end": len(best_policy) if best_policy else 0,
            "answer": str(best_policy) 
        }