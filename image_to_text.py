import logging
from typing import Any, Dict, List, Union, Optional
from .base import Pipeline

# Core Project Dependencies
from core.workflow import Workflow
from core.environment import Environment

logger = logging.getLogger(__name__)

class OffloadingImageToTextPipeline(Pipeline):
    """
    Captioning pipeline for offloading strategies.
    Predicts a strategic caption for a given offloading scenario.
    
    Analogy to ImageToTextPipeline:
    - Input Image -> Workflow DAG & Environment matrices
    - Prompt -> Optimization Objective (e.g., 'Target Latency')
    - Generated Text -> Strategic Policy Summary
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Check if the model can generate captions for offloading policies
        self.check_model_type("OFFLOADING_CAPTIONER")

    def _sanitize_parameters(self, max_new_tokens=None, prompt=None, timeout=None, **kwargs):
        """Standardized parameter sanitizer."""
        preprocess_params = {}
        forward_params = {}

        if prompt is not None:
            preprocess_params["prompt"] = prompt
        if timeout is not None:
            preprocess_params["timeout"] = timeout
        if max_new_tokens is not None:
            forward_params["max_new_tokens"] = max_new_tokens

        return preprocess_params, forward_params, {}

    def preprocess(self, inputs: Dict, prompt: str = None, timeout: float = None) -> Dict:
        """
        Step 1: Scenario Loading.
        Prepares the DAG and Environment for strategic description.
        """
        workflow_obj = Workflow.from_experiment_dict(inputs.get("workflow", {}))
        
        # If a prompt is provided, we use it for conditional strategy generation
        # mimicking 'pix2struct' or 'git' conditional captioning.
        return {
            "workflow": workflow_obj,
            "env": inputs.get("env", {}),
            "objective": prompt or "general optimization",
            "params": inputs.get("costs", {})
        }

    def _forward(self, model_inputs: Dict, **generate_kwargs) -> Dict:
        """
        Step 2: Strategy Generation.
        The Planner and Output agents generate the 'caption'.
        """
        # Call Planner to generate the core strategy text
        plan = self.model.planner.create_plan(
            model_inputs["env"], 
            model_inputs["workflow"].to_experiment_dict(),
            model_inputs["params"]
        )

        # Call Evaluator to ground the strategy in a policy vector
        result = self.model.evaluator.find_best_policy(
            model_inputs["workflow"].to_experiment_dict(),
            model_inputs["env"],
            model_inputs["params"],
            plan=plan
        )
        
        return {
            "strategy_text": plan,
            "policy": result["best_policy"],
            "objective": model_inputs["objective"]
        }

    def postprocess(self, model_outputs: Dict) -> List[Dict[str, str]]:
        """
        Step 3: Caption Formatting.
        Returns the generated strategic text.
        """
        # Mirrors the 'records' structure of image_to_text.py
        #
        return [{
            "generated_text": model_outputs["strategy_text"],
            "policy_vector": list(model_outputs["policy"]) if model_outputs["policy"] else []
        }]