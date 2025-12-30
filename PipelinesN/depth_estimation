import logging
from typing import Any, Dict, List, Union, overload
from .base import Pipeline

# Core project imports
from core.workflow import Workflow
from core.environment import Environment

logger = logging.getLogger(__name__)

class OffloadingEstimationPipeline(Pipeline):
    """
    Offloading estimation pipeline. This pipeline predicts the performance 'depth'
    (cost, latency, energy) of a task workflow across an edge-cloud landscape.
    
    Analogy to DepthEstimationPipeline:
    - Pixel Depth (Meters) -> Task Cost (Utility Units)
    - Depth Image (PIL) -> Performance Report (Visual/Text)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure the model is compatible with our offloading reasoning engine
        self.check_model_type("AGENTIC_OFFLOADING")

    def _sanitize_parameters(self, timeout=None, **kwargs):
        """Sanitize parameters for offloading context."""
        preprocess_params = {}
        if timeout is not None:
            preprocess_params["timeout"] = timeout
        return preprocess_params, {}, {}

    def preprocess(self, scenario_data: Dict, timeout=None):
        """
        Step 1: Context Loading.
        Prepares the DAG and Environment, setting the 'target_size' (number of tasks).
        """
        workflow_obj = Workflow.from_experiment_dict(scenario_data.get("workflow", {}))
        
        # We store the 'target_size' as the number of tasks, similar to image dimensions
        #
        return {
            "workflow": workflow_obj,
            "env": scenario_data.get("env", {}),
            "params": scenario_data.get("costs", {}),
            "target_size": workflow_obj.N 
        }

    def _forward(self, model_inputs):
        """
        Step 2: Strategic Prediction.
        The Planner and Evaluator agents determine the 'depth' (cost) of each placement.
        """
        target_size = model_inputs.pop("target_size")
        
        # Call Planner for strategic analysis
        plan = self.model.planner.create_plan(
            model_inputs["env"], 
            model_inputs["workflow"].to_experiment_dict(), 
            model_inputs["params"]
        )
        
        # Call Evaluator to find the 'depth' of the optimal policy
        eval_result = self.model.evaluator.find_best_policy(
            model_inputs["workflow"].to_experiment_dict(),
            model_inputs["env"],
            model_inputs["params"],
            plan=plan
        )
        
        return {
            "best_policy": eval_result["best_policy"],
            "best_cost": eval_result["best_cost"],
            "target_size": target_size,
            "strategic_plan": plan
        }

    def postprocess(self, model_outputs):
        """
        Step 3: Output Formatting.
        Normalizes the raw cost and formats the final policy mapping.
        """
        policy = model_outputs["best_policy"]
        cost = model_outputs["best_cost"]
        
        # Normalization analogy: In depth estimation, depth is normalized to [0, 255]
        # for visualization. Here we represent it as a relative efficiency score.
        #
        efficiency_score = 1.0 / (1.0 + cost) if cost != float('inf') else 0.0

        formatted_output = {
            "predicted_policy": policy,
            "utility_cost": cost,
            "efficiency_score": efficiency_score,
            "strategic_rationale": model_outputs["strategic_plan"]
        }

        return formatted_output