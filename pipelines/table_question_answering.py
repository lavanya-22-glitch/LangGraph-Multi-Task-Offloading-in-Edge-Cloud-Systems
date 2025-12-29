import collections
import types
import warnings

import numpy as np

# --- EDGE COMPUTING CORE IMPORTS ---
from core.workflow import Workflow
from core.environment import Environment
from agents.planner import PlannerAgent
from agents.evaluator import EvaluatorAgent

# Mocking internal Hugging Face Utils to match reference structure
class ArgumentHandler:
    """Mock ArgumentHandler for compatibility."""
    pass

class Pipeline:
    """Mock Base Pipeline for compatibility."""
    def __init__(self, **kwargs):
        pass
    def __call__(self, inputs, **kwargs):
        return inputs
    def check_model_type(self, mapping):
        pass

def add_end_docstrings(*args):
    def decorator(func):
        return func
    return decorator

def build_pipeline_init_args(**kwargs):
    return ""

def is_torch_available():
    return False

def requires_backends(obj, backend):
    pass

class TableQuestionAnsweringArgumentHandler(ArgumentHandler):
    """
    Handles arguments for the TableQuestionAnsweringPipeline.
    Normalizes inputs into a list of dicts: [{"table": ..., "query": ...}]
    """

    def __call__(self, table=None, query=None, **kwargs):
        # Returns tqa_pipeline_inputs of shape:
        # [
        #   {"table": pd.DataFrame, "query": list[str]},
        #   ...,
        #   {"table": pd.DataFrame, "query" : list[str]}
        # ]
        # In our case, 'table' is the Environment Dict, and 'query' is the Task Description.
        
        if table is None:
            raise ValueError("Keyword argument `table` cannot be None.")
        elif query is None:
            if isinstance(table, dict) and table.get("query") is not None and table.get("table") is not None:
                tqa_pipeline_inputs = [table]
            elif isinstance(table, list) and len(table) > 0:
                tqa_pipeline_inputs = table
            else:
                 tqa_pipeline_inputs = [{"table": table, "query": "Optimize Offloading"}] # Default query
        else:
            tqa_pipeline_inputs = [{"table": table, "query": query}]

        return tqa_pipeline_inputs


@add_end_docstrings(build_pipeline_init_args(has_tokenizer=True))
class TableQuestionAnsweringPipeline(Pipeline):
    """
    Table Question Answering pipeline using `PlannerAgent`. 
    
    In this Edge Computing context:
    - **Table**: The Environment constraints (Bandwidth, Power, CPU specs).
    - **Query**: The Workflow requirements (Tasks, Deadlines).
    - **Answer**: The specific resource allocation values derived from the policy.

    Example:

    ```python
    >>> from pipelines import pipeline
    >>> oracle = pipeline(task="table-question-answering")
    >>> env_data = {"bandwidth": "50MB/s", "cloud_cpu": "3.0GHz"}
    >>> oracle(query="Allocate resources for high-priority tasks", table=env_data)
    {'answer': 'AGGREGATOR > CLOUD', 'coordinates': [(0, 1)], 'cells': ['3.0GHz']}
    ```
    """

    default_input_names = "table,query"

    _pipeline_calls_generate = True
    _load_processor = False
    _load_image_processor = False
    _load_feature_extractor = False
    _load_tokenizer = True

    def __init__(self, api_key: str, args_parser=TableQuestionAnsweringArgumentHandler(), **kwargs):
        super().__init__(**kwargs)
        self._args_parser = args_parser
        
        # Initialize Agents
        self.planner = PlannerAgent(api_key=api_key)
        self.evaluator = EvaluatorAgent(api_key=api_key)

        # Mocking model config
        self.type = "tapas" 
        self.aggregate = True 

    def __call__(self, *args, **kwargs):
        """
        Answers queries according to a table (Environment).
        """
        pipeline_inputs = self._args_parser(*args, **kwargs)

        # We treat the list handling manually here since we are mocking super().__call__
        results = []
        for input_data in pipeline_inputs:
             # 1. Preprocess
             model_inputs = self.preprocess(input_data, **kwargs)
             
             # 2. Forward
             model_outputs = self._forward(model_inputs)
             
             # 3. Postprocess
             result = self.postprocess(model_outputs)
             results.append(result)

        if len(results) == 1:
            return results[0]
        return results

    def _sanitize_parameters(self, sequential=None, padding=None, truncation=None, **kwargs):
        preprocess_params = {}
        if padding is not None:
            preprocess_params["padding"] = padding
        if truncation is not None:
            preprocess_params["truncation"] = truncation

        forward_params = {}
        if sequential is not None:
            forward_params["sequential"] = sequential

        return preprocess_params, forward_params, {}

    def preprocess(self, pipeline_input, padding=True, truncation=None):
        """
        Prepare the Environment (Table) and Task (Query) for the Agent.
        """
        table, query = pipeline_input["table"], pipeline_input["query"]
        
        # We assume 'table' comes in as the Environment Dictionary
        # We assume 'query' comes in as the Workflow Dictionary (or string description)
        
        if table is None:
            raise ValueError("table is empty")
        if query is None:
            raise ValueError("query is empty")
            
        return {
            "env": table,     # The "Table"
            "workflow": query # The "Query"
        }

    def _forward(self, model_inputs, sequential=False, **generate_kwargs):
        """
        The Planner analyzes the Table (Env) and Query (Workflow).
        """
        # Ensure we have a valid workflow dict. If query was a string, we might need a dummy workflow or parsing.
        # For this implementation, we assume the user passes the Workflow Dict as the 'query'.
        workflow = model_inputs["workflow"]
        env = model_inputs["env"]
        
        # If workflow is just a string description, we rely on the planner to interpret it genericly
        # But create_plan expects a dict.
        if isinstance(workflow, str):
             # Dummy wrapper if strictly text passed
             workflow = {"description": workflow, "N": 5} 

        plan = self.planner.create_plan(
            workflow, env, []
        )
        
        return {
            "model_inputs": model_inputs,
            "table": env,
            "outputs": plan # The text plan is our "logits"
        }

    def postprocess(self, model_outputs):
        """
        Extract specific answers from the Planner's strategy.
        """
        inputs = model_outputs["model_inputs"]
        env = model_outputs["table"]
        plan = model_outputs["outputs"]
        
        # Use Evaluator to calculate the concrete "Answer" (The Cost/Policy)
        result = self.evaluator.find_best_policy(
             inputs["workflow"] if isinstance(inputs["workflow"], dict) else {"N":5}, # Handle dummy
             env,
             params={},
             plan=plan
        )
        
        best_policy = result["best_policy"]
        best_cost = result["best_cost"]
        
        # Format exactly like TableQA output
        # coordinates: (row, col) -> (task_id, server_id)
        coordinates = []
        cells = []
        
        if best_policy:
            for task_idx, server_id in enumerate(best_policy):
                coordinates.append((task_idx, server_id))
                # The "Cell" value is the placement decision
                cells.append("CLOUD" if server_id == 1 else "EDGE")

        answer = {
            "answer": f"OPTIMAL_COST > {best_cost:.4f}",
            "coordinates": coordinates,
            "cells": cells,
            "aggregator": "MINIMIZE_COST"
        }

        return answer