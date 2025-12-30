import enum
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Union, overload
from .base import Pipeline  # Base HF Pipeline

logger = logging.getLogger(__name__)

class ReturnType(enum.Enum):
    """Matches the ReturnType enum in any_to_any.py"""
    TENSORS = 0      # Return raw binary policy
    NEW_TEXT = 1     # Return only the new strategic plan
    FULL_RESULTS = 2 # Return the full execution trace

class OffloadingAnyToAnyPipeline(Pipeline):
    """
    Multimodal Offloading Pipeline. 
    Accepts DAG text, Environment dicts, or historical Memory Chats.
    """
    _load_processor = True
    
    def _sanitize_parameters(
        self,
        max_new_tokens=None,
        return_full_text=None,
        return_type=None,
        delta_t=1,
        delta_e=1,
        **kwargs
    ):
        """Standardized parameter sanitizer."""
        preprocess_params = kwargs
        forward_kwargs = {"generate_kwargs": {}}
        postprocess_params = {"delta_t": delta_t, "delta_e": delta_e}

        if max_new_tokens is not None:
            forward_kwargs["generate_kwargs"]["max_new_tokens"] = max_new_tokens

        # Mutually exclusive logic for return types
        if return_full_text is not None and return_type is None:
            return_type = ReturnType.FULL_RESULTS if return_full_text else ReturnType.NEW_TEXT
        
        if return_type is not None:
            postprocess_params["return_type"] = return_type

        return preprocess_params, forward_kwargs, postprocess_params

    def preprocess(self, inputs=None, **processing_kwargs):
        """
        Prepares the DAG and Environment for the Planner.
        Supports 'Chat' format for Few-Shot memory learning.
        """
        # If input is a list of dicts (Chat format), process as historical memory
        if isinstance(inputs, list) and len(inputs) > 0 and "role" in inputs[0]:
            # Convert to internal Chat object similar to any_to_any.py
            return {"chat_history": inputs, "mode": "memory_augmented"}

        # Standard dictionary input (Workflow + Env)
        from ..core.workflow import Workflow
        workflow_obj = Workflow.from_experiment_dict(inputs.get("workflow", {}))
        
        return {
            "workflow": workflow_obj,
            "env": inputs.get("env", {}),
            "params": inputs.get("costs", {}),
            "raw_text": str(inputs.get("workflow"))
        }

    def _forward(self, model_inputs, generate_kwargs=None):
        """
        The 'Inference' pass where the Planner Agent analyzes the DAG.
        Matches the _forward structure of any_to_any.py.
        """
        # Run the Planner Agent (Strategic Analysis)
        # This replaces the 'self.model.generate' call in any_to_any.py
        plan = self.model.planner.create_plan(
            model_inputs["env"],
            model_inputs["workflow"].to_experiment_dict(),
            model_inputs["params"]
        )

        return {
            "generated_plan": plan,
            "workflow": model_inputs["workflow"],
            "env": model_inputs["env"],
            "params": model_inputs["params"]
        }

    def postprocess(self, model_outputs, return_type=ReturnType.FULL_RESULTS, **kwargs):
        """
        Decodes the Planner's text into an optimal binary policy.
        """
        # Run the Evaluator Agent (Equivalent to multimodal decoding)
        eval_result = self.model.evaluator.find_best_policy(
            model_outputs["workflow"].to_experiment_dict(),
            model_outputs["env"],
            model_outputs["params"],
            plan=model_outputs["generated_plan"]
        )

        # Logic for determining what to return based on ReturnType
        if return_type == ReturnType.TENSORS:
            return {"generated_policy": eval_result["best_policy"]}

        # Construct the final record
        record = {
            "input_workflow": str(model_outputs["workflow"]),
            "generated_plan": model_outputs["generated_plan"],
            "optimal_policy": eval_result["best_policy"],
            "utility_cost": eval_result["best_cost"]
        }

        return [record] if return_type == ReturnType.FULL_RESULTS else record