import numpy as np
import torch
import logging
from typing import Any, Dict, List, Union, Tuple
from .base import ChunkPipeline # Inheriting from ChunkPipeline for stride support

# Core Dependencies
from core.workflow import Workflow
from core.environment import Environment

logger = logging.getLogger(__name__)

class ChunkedOffloadingPipeline(ChunkPipeline):
    """
    Pipeline for long-form offloading optimization.
    It splits a massive DAG into chunks, optimizes them individually with stride context,
    and merges the placement policies.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Default stride parameters for graph context
        self.chunk_length_tasks = kwargs.get("chunk_length_tasks", 10)
        self.stride_tasks = kwargs.get("stride_tasks", 2)

    def _sanitize_parameters(self, chunk_length_tasks=None, stride_tasks=None, **kwargs):
        """Standardized parameter sanitizer."""
        preprocess_params = {}
        if chunk_length_tasks is not None:
            preprocess_params["chunk_length_tasks"] = chunk_length_tasks
        if stride_tasks is not None:
            preprocess_params["stride_tasks"] = stride_tasks
            
        return preprocess_params, kwargs, {}

    def preprocess(self, inputs: Dict, chunk_length_tasks=10, stride_tasks=2):
        """
        Equivalent to ASR chunking. Breaks a massive DAG into windows.
        """
        workflow_dict = inputs.get("workflow", {})
        env_dict = inputs.get("env", {})
        
        # Build full workflow
        full_wf = Workflow.from_experiment_dict(workflow_dict)
        tasks = full_wf.vertices()[1:-1] # Exclude entry/exit
        
        # Chunking Logic: Sliding window over task IDs
        step = chunk_length_tasks - (2 * stride_tasks)
        for i in range(0, len(tasks), step):
            chunk_tasks = tasks[i : i + chunk_length_tasks]
            is_last = (i + chunk_length_tasks) >= len(tasks)
            
            # Create a sub-graph for this chunk
            sub_wf = self._extract_subgraph(full_wf, chunk_tasks)
            
            # Define stride context (what to ignore in the final result)
            stride = (len(chunk_tasks), 
                      0 if i == 0 else stride_tasks, 
                      0 if is_last else stride_tasks)
            
            yield {
                "is_last": is_last,
                "stride": stride,
                "workflow": sub_wf.to_experiment_dict(),
                "env": env_dict,
                "params": inputs.get("costs", {})
            }

    def _forward(self, model_inputs, **generate_kwargs):
        """Strategic planning for a single chunk."""
        stride = model_inputs.pop("stride")
        is_last = model_inputs.pop("is_last")
        
        # Call Planner and Evaluator for this chunk
        plan = self.model.planner.create_plan(model_inputs["env"], model_inputs["workflow"], model_inputs["params"])
        result = self.model.evaluator.find_best_policy(model_inputs["workflow"], model_inputs["env"], model_inputs["params"], plan=plan)
        
        return {
            "is_last": is_last,
            "stride": stride,
            "tokens": torch.tensor(result["best_policy"]), # Policies as 'tokens'
            "plan": plan
        }

    def postprocess(self, model_outputs):
        """
        Reconciles overlapping policies by discarding the stride bits.
        """
        final_policy = []
        for output in model_outputs:
            policy = output["tokens"].numpy()
            total_n, left, right = output["stride"]
            
            # Slice the policy to remove stride context
            right_n = total_n - right
            valid_segment = policy[left:right_n]
            final_policy.extend(valid_segment.tolist())
            
        return {"optimal_policy": final_policy}

    def _extract_subgraph(self, full_wf, task_ids):
        """Helper to create a valid sub-DAG for a chunk."""
        # Custom logic to isolate tasks and maintain internal edges
        pass