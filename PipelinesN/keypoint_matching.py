import logging
from typing import Any, Dict, List, Union, TypedDict
from .base import Pipeline

# Core Dependencies
from core.workflow import Workflow
from core.environment import Environment

logger = logging.getLogger(__name__)

class TaskKeypoint(TypedDict):
    """Matches the Keypoint structure in keypoint_matching.py."""
    task_id: int
    v_i: float  # Computational load as a 'coordinate' feature

class DependencyMatch(TypedDict):
    """Matches the Match structure in keypoint_matching.py."""
    task_source: TaskKeypoint
    task_target: TaskKeypoint
    score: float  # Matching confidence based on utility cost delta

class OffloadingKeypointMatchingPipeline(Pipeline):
    """
    Keypoint matching pipeline for offloading scenarios.
    Matches dependencies between two different workflow configurations.
    
    Analogy to KeypointMatchingPipeline:
    - Image Pair -> Two different Environment/Workflow scenarios
    - Keypoints -> Critical Path tasks and their dependencies
    - Matching Scores -> Relative cost savings of migrating specific task links
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Verify model capabilities for dependency matching
        self.check_model_type("OFFLOADING_KEYPOINT_MATCHER")

    def _sanitize_parameters(self, threshold=0.0, **kwargs):
        """Standardized parameter sanitizer."""
        preprocess_params = {}
        postprocess_params = {"threshold": threshold}
        return preprocess_params, {}, postprocess_params

    def preprocess(self, scenario_pair: List[Dict], **kwargs) -> Dict:
        """
        Equivalent to image pair loading. Prepares two scenarios for matching.
        """
        # We expect a pair of scenarios (Source and Target)
        images = [scenario_pair[0], scenario_pair[1]]
        
        # Load workflows for both scenarios
        wf_source = Workflow.from_experiment_dict(images[0].get("workflow", {}))
        wf_target = Workflow.from_experiment_dict(images[1].get("workflow", {}))
        
        return {
            "wf_source": wf_source,
            "wf_target": wf_target,
            "env_source": images[0].get("env", {}),
            "env_target": images[1].get("env", {}),
            "params": images[0].get("costs", {})
        }

    def _forward(self, model_inputs: Dict) -> Dict:
        """
        Step 2: Forward Pass.
        Identifies corresponding bottleneck tasks across the two environments.
        """
        # Call Planner to analyze source and target strategies
        source_plan = self.model.planner.create_plan(
            model_inputs["env_source"], model_inputs["wf_source"].to_experiment_dict(), model_inputs["params"]
        )
        target_plan = self.model.planner.create_plan(
            model_inputs["env_target"], model_inputs["wf_target"].to_experiment_dict(), model_inputs["params"]
        )
        
        return {
            "source_tasks": model_inputs["wf_source"].vertices(),
            "target_tasks": model_inputs["wf_target"].vertices(),
            "source_v": model_inputs["wf_source"].V(),
            "target_v": model_inputs["wf_target"].V(),
            "confidence_scores": [1.0] * len(model_inputs["wf_source"].vertices()) # Dummy scores
        }

    def postprocess(self, model_outputs: Dict, threshold=0.0) -> List[DependencyMatch]:
        """
        Step 3: Postprocessing.
        Filters and formats the matches based on a confidence threshold.
        """
        matches = []
        for i, (t_s, t_t) in enumerate(zip(model_outputs["source_tasks"], model_outputs["target_tasks"])):
            # Analogy to score thresholding in keypoint_matching.py
            if model_outputs["confidence_scores"][i] >= threshold:
                kp_s = TaskKeypoint(task_id=t_s, y=model_outputs["source_v"].get(t_s, 0))
                kp_t = TaskKeypoint(task_id=t_t, y=model_outputs["target_v"].get(t_t, 0))
                
                matches.append(DependencyMatch(
                    keypoint_image_0=kp_s,
                    keypoint_image_1=kp_t,
                    score=model_outputs["confidence_scores"][i]
                ))
        
        return matches