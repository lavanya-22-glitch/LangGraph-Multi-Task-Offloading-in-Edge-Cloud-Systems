import numpy as np
from typing import Any, Dict, List, Union, Optional
from .base import Pipeline

# Core Project Dependencies
from core.workflow import Workflow
from core.environment import Environment

class OffloadingRefinementPipeline(Pipeline):
    """
    Refinement pipeline for task offloading policies.
    Transforms a sub-optimal 'raw' policy into an optimized 'reconstructed' policy.
    
    Analogy to ImageToImagePipeline:
    - Input Image -> Initial Placement Policy
    - Reconstruction -> Refined placement with lower utility cost
    - Image Normalization -> Cost scaling (0 to 1 range)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Verify that the model is equipped for iterative refinement tasks
        self.check_model_type("OFFLOADING_REFINER")

    def _sanitize_parameters(self, **kwargs):
        """Standardized parameter sanitizer."""
        preprocess_params = {}
        if "timeout" in kwargs:
            preprocess_params["timeout"] = kwargs["timeout"]
        return preprocess_params, {}, {}

    def preprocess(self, inputs: Dict, timeout: Optional[float] = None) -> Dict:
        """
        Step 1: Context Preparation.
        Prepares the original DAG and the current 'noisy' policy for refinement.
        """
        workflow_obj = Workflow.from_experiment_dict(inputs.get("workflow", {}))
        current_policy = inputs.get("initial_policy", [])
        
        return {
            "workflow": workflow_obj,
            "env": inputs.get("env", {}),
            "initial_policy": current_policy,
            "params": inputs.get("costs", {})
        }

    def _forward(self, model_inputs: Dict) -> Dict:
        """
        Step 2: Strategic Reconstruction.
        The Planner and Evaluator agents work to improve the initial strategy.
        """
        # Call Planner to identify bottlenecks in the current policy
        refinement_plan = self.model.planner.create_plan(
            model_inputs["env"], 
            model_inputs["workflow"].to_experiment_dict(),
            {**model_inputs["params"], "refine_target": model_inputs["initial_policy"]}
        )
        
        # Use the Evaluator to find a strictly better policy
        refinement_result = self.model.evaluator.find_best_policy(
            model_inputs["workflow"].to_experiment_dict(),
            model_inputs["env"],
            model_inputs["params"],
            plan=refinement_plan
        )
        
        return {
            "reconstruction": refinement_result["best_policy"],
            "original_cost": model_inputs["params"].get("initial_cost", float('inf')),
            "new_cost": refinement_result["best_cost"]
        }

    def postprocess(self, model_outputs: Dict) -> Union[List[int], Dict]:
        """
        Step 3: Policy Formatting.
        Returns the refined placement vector.
        """
        # Reconstruction analogy: we treat the placement vector as the 'image' output
        #
        refined_policy = model_outputs["reconstruction"]
        
        # Return improvement statistics similar to reconstruction quality metrics
        return {
            "refined_policy": list(refined_policy) if refined_policy else [],
            "improvement_delta": model_outputs["original_cost"] - model_outputs["new_cost"],
            "is_improved": model_outputs["new_cost"] < model_outputs["original_cost"]
        }