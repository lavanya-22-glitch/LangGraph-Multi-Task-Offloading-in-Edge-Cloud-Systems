import numpy as np
from typing import Any, Dict, List, Union, Optional
from .base import Pipeline

# Core Dependencies
from core.workflow import Workflow
from core.environment import Environment

class OffloadingSegmentationPipeline(Pipeline):
    """
    Segmentation pipeline for offloading scenarios.
    Partitions a DAG into execution clusters (masks) based on location assignment.
    
    Analogy to ImageSegmentationPipeline:
    - Image Input -> Workflow DAG
    - Object Masks -> Clusters of tasks assigned to the same node
    - Class Labels -> Location Types (IoT, Edge, Cloud)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure the underlying agents are ready for partitioning tasks
        self.check_model_type("OFFLOADING_SEGMENTER")

    def _sanitize_parameters(self, threshold=None, **kwargs):
        """Standardized parameter sanitizer for segmentation context."""
        preprocess_params = {}
        postprocess_params = {}
        if threshold is not None:
            postprocess_params["threshold"] = threshold
        if "timeout" in kwargs:
            preprocess_params["timeout"] = kwargs["timeout"]
            
        return preprocess_params, {}, postprocess_params

    def preprocess(self, inputs: Dict, timeout: Optional[float] = None) -> Dict:
        """
        Equivalent to image processing. Prepares the DAG for segment prediction.
        """
        workflow_obj = Workflow.from_experiment_dict(inputs.get("workflow", {}))
        target_size = workflow_obj.N
        
        return {
            "workflow": workflow_obj,
            "env": inputs.get("env", {}),
            "params": inputs.get("costs", {}),
            "target_size": target_size
        }

    def _forward(self, model_inputs: Dict) -> Dict:
        """
        Uses the Planner and Evaluator to find the segment boundaries (placement policy).
        """
        target_size = model_inputs.pop("target_size")
        
        # Strategic analysis for partitioning
        plan = self.model.planner.create_plan(
            model_inputs["env"], 
            model_inputs["workflow"].to_experiment_dict(),
            model_inputs["params"]
        )
        
        # Determine the final policy p*
        result = self.model.evaluator.find_best_policy(
            model_inputs["workflow"].to_experiment_dict(),
            model_inputs["env"],
            model_inputs["params"],
            plan=plan
        )
        
        return {
            "policy": result["best_policy"],
            "score": result["best_cost"],
            "target_size": target_size,
            "env": model_inputs["env"]
        }

    def postprocess(self, model_outputs: Dict, threshold: float = 0.9) -> List[Dict]:
        """
        Converts the raw policy into labeled 'segments' (clusters of tasks).
        """
        policy = model_outputs["policy"]
        locations = model_outputs["env"].get("locations", {})
        
        # Group tasks by their assigned location (Segmenting the DAG)
        #
        segments = {}
        for task_idx, loc_id in enumerate(policy, start=1):
            if loc_id not in segments:
                segments[loc_id] = []
            segments[loc_id].append(task_idx)
            
        annotations = []
        for loc_id, task_list in segments.items():
            label = locations.get(loc_id, "unknown").upper()
            
            # Create a segmentation record mirroring image_segmentation.py output
            annotations.append({
                "score": model_outputs["score"], # Utility cost for this whole segmentation
                "label": label,
                "mask": task_list, # The list of tasks identifying this 'object' in the DAG
                "location_id": loc_id
            })
            
        return annotations