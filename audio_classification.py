import numpy as np
import logging
from typing import Any, Dict, List, Union
from .base import Pipeline

# Dependencies from your core project
from core.workflow import Workflow
from core.environment import Environment

logger = logging.getLogger(__name__)

class OffloadingClassificationPipeline(Pipeline):
    """
    Task offloading optimization pipeline. This pipeline predicts the optimal 
    placement policy p* for a raw DAG workflow. 
    
    Analogy to AudioClassificationPipeline:
    - Audio Waveform -> Workflow DAG
    - Sampling Rate -> CCR (Communication-to-Computation Ratio)
    - Resampling -> Feature Normalization (Units scaling)
    """

    def __init__(self, *args, **kwargs):
        # Default top_k identifies how many candidate policies to return
        if "top_k" not in kwargs:
            kwargs["top_k"] = 5
        super().__init__(*args, **kwargs)

    def _sanitize_parameters(self, top_k=None, function_to_apply=None, **kwargs):
        """Standardized parameter sanitizer."""
        postprocess_params = {}
        
        # Mapping top_k logic from audio classification
        postprocess_params["top_k"] = top_k if top_k is not None else 5
        
        # In offloading, function_to_apply determines the normalization of the cost U(w,p)
        postprocess_params["function_to_apply"] = function_to_apply or "softmax"
        
        return {}, {}, postprocess_params

    def preprocess(self, inputs: Union[Dict, str]):
        """
        Equivalent to audio resampling. Normalizes MI/Bytes into a common 
        tensor format for the agents.
        """
        if isinstance(inputs, str):
            # Load local scenario file if string is provided
            import json
            with open(inputs, "r") as f:
                inputs = json.load(f)

        if isinstance(inputs, dict):
            # RESAMPLING ANALOGY: Ensure units match the core environment expectations
            inputs = inputs.copy()
            workflow_obj = Workflow.from_experiment_dict(inputs.get("workflow", {}))
            
            # If the input DAG units (MI) don't match the internal 'sampling rate' 
            # (CPU cycles), perform a conversion.
            if inputs.get("unit") == "MI":
                for task in workflow_obj._tasks:
                    task.v_i *= 1e6 # "Resample" MI to Cycles
            
            return {
                "workflow": workflow_obj,
                "env": inputs.get("env", {}),
                "params": inputs.get("costs", {})
            }

        raise TypeError("Input must be a scenario dictionary or a path to a JSON file")

    def _forward(self, model_inputs):
        """
        Executes the agentic reasoning pass.
        """
        # Call the Planner Agent (Reasoning)
        plan = self.model.planner.create_plan(
            model_inputs["env"],
            model_inputs["workflow"].to_experiment_dict(),
            model_inputs["params"]
        )
        
        # Call the Evaluator Agent (Search)
        eval_result = self.model.evaluator.find_best_policy(
            model_inputs["workflow"].to_experiment_dict(),
            model_inputs["env"],
            model_inputs["params"],
            plan=plan
        )
        
        return {"eval_result": eval_result, "plan": plan}

    def postprocess(self, model_outputs, top_k=5, function_to_apply="softmax"):
        """
        Returns the top_k policies with their corresponding scores.
        """
        result = model_outputs["eval_result"]
        
        # Return the best policy in a structured format matching audio classification
        return [
            {
                "label": f"Policy {i+1}",
                "score": result["best_cost"],
                "policy": result["best_policy"],
                "explanation": model_outputs["plan"]
            }
        ]