import enum
import logging
from typing import Any, Dict, List, Optional, Union
from .base import Pipeline

# Core Dependencies
from core.workflow import Workflow
from core.environment import Environment

logger = logging.get_logger(__name__)

class ReturnType(enum.Enum):
    """Matches the ReturnType enum in image_text_to_text.py."""
    TENSORS = 0     # Return raw policy vector
    NEW_TEXT = 1    # Return only the new strategic justification
    FULL_TEXT = 2   # Return the query + justification

class OffloadingImageTextToTextPipeline(Pipeline):
    """
    Multimodal offloading pipeline for generating strategic justifications.
    
    Analogy to ImageTextToTextPipeline:
    - Image Input -> DAG Structure & Environment Matrices
    - Text Input -> Optimization Query (e.g., 'Minimize Energy')
    - Chat Format -> Multi-turn optimization refinement
    """

    def _sanitize_parameters(
        self,
        return_full_text=None,
        return_type=None,
        continue_final_message=None,
        **kwargs
    ):
        """Standardized parameter sanitizer."""
        preprocess_params = kwargs
        if continue_final_message is not None:
            preprocess_params["continue_final_message"] = continue_final_message

        postprocess_params = {}
        if return_full_text is not None and return_type is None:
            return_type = ReturnType.FULL_TEXT if return_full_text else ReturnType.NEW_TEXT
        
        if return_type is not None:
            postprocess_params["return_type"] = return_type

        return preprocess_params, {}, postprocess_params

    def preprocess(self, inputs: Dict, continue_final_message: bool = None, **kwargs) -> Dict:
        """
        Equivalent to multimodal feature fusion. Combines the 'image' (scenario) 
        with the 'text' (query).
        """
        workflow_obj = Workflow.from_experiment_dict(inputs.get("workflow", {}))
        
        # If the input ends with an 'assistant' message, we enable prefill mode
        # mimicking conversational vision models.
        is_prefill = continue_final_message or inputs.get("mode") == "assistant"

        return {
            "workflow": workflow_obj,
            "env": inputs.get("env", {}),
            "query": inputs.get("text", "Optimize this scenario."),
            "params": inputs.get("costs", {}),
            "is_prefill": is_prefill
        }

    def _forward(self, model_inputs: Dict) -> Dict:
        """
        Generates the 'sequence' (policy + justification) using agentic reasoning.
        """
        # Planner acts as the multimodal decoder
        plan_result = self.model.planner.create_plan(
            model_inputs["env"], 
            model_inputs["workflow"].to_experiment_dict(),
            model_inputs["params"]
        )

        # Evaluator provides the 'token' (policy) grounding
        eval_result = self.model.evaluator.find_best_policy(
            model_inputs["workflow"].to_experiment_dict(),
            model_inputs["env"],
            model_inputs["params"],
            plan=plan_result
        )

        return {
            "justification": plan_result,
            "policy": eval_result["best_policy"],
            "query": model_inputs["query"],
            "is_prefill": model_inputs["is_prefill"]
        }

    def postprocess(self, model_outputs: Dict, return_type=ReturnType.FULL_TEXT) -> Dict:
        """
        Formats the output according to the requested ReturnType.
        """
        query = model_outputs["query"]
        justification = model_outputs["justification"]
        
        if return_type == ReturnType.TENSORS:
            return {"policy_vector": model_outputs["policy"]}

        # Logic for 'NEW_TEXT' vs 'FULL_TEXT'
        generated_text = justification
        if return_type == ReturnType.FULL_TEXT:
            generated_text = f"Query: {query}\nStrategy: {justification}"

        return {
            "generated_text": generated_text,
            "policy": model_outputs["policy"]
        }