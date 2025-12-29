from typing import Any, Dict, List, Union, overload
import logging

# --- EDGE COMPUTING CORE IMPORTS ---
from core.workflow import Workflow
from agents.planner import PlannerAgent
from agents.evaluator import EvaluatorAgent

logger = logging.getLogger(__name__)

class ObjectDetectionPipeline:
    """
    Object detection pipeline using `PlannerAgent`. This pipeline predicts 'bounding boxes' 
    (clusters of tasks) that form critical paths and assigns them to resources.

    Example:

    ```python
    >>> from pipelines import pipeline
    >>> detector = pipeline(task="bottleneck-detection")
    >>> detector(workflow_data)
    [{'score': 0.99, 'label': 'CRITICAL_PATH', 'box': {'xmin': 1, 'ymin': 1, 'xmax': 5, 'ymax': 1}}]
    ```
    """

    _load_processor = False
    _load_image_processor = True
    _load_feature_extractor = False
    _load_tokenizer = None

    def __init__(self, api_key: str, **kwargs):
        self.planner = PlannerAgent(api_key=api_key)
        self.evaluator = EvaluatorAgent(api_key=api_key)
        
        # Mocking model config
        self.model = type('MockModel', (), {'config': type('Config', (), {'id2label': {1: 'CLOUD', 0: 'LOCAL'}})})

    def _sanitize_parameters(self, **kwargs):
        preprocess_params = {}
        if "timeout" in kwargs:
            preprocess_params["timeout"] = kwargs["timeout"]
        postprocess_kwargs = {}
        if "threshold" in kwargs:
            postprocess_kwargs["threshold"] = kwargs["threshold"]
        return preprocess_params, {}, postprocess_kwargs

    def __call__(self, inputs: Union[Dict, Any], **kwargs) -> List[Dict[str, Any]]:
        """
        Detect bottlenecks (bounding boxes) in the workflow graph.
        """
        preprocess_params, _, postprocess_kwargs = self._sanitize_parameters(**kwargs)
        
        # 1. Preprocess
        model_inputs = self.preprocess(inputs, **preprocess_params)
        
        # 2. Forward
        model_outputs = self._forward(model_inputs)
        
        # 3. Postprocess
        return self.postprocess(model_outputs, **postprocess_kwargs)

    def preprocess(self, inputs, timeout=None):
        """
        Load workflow and identify potential critical paths (Edges).
        """
        # Replaces load_image
        workflow = inputs.get("workflow")
        env = inputs.get("env")
        
        target_size = workflow.get("N", 0) # Rough equivalent to image size
        
        return {
            "workflow": workflow,
            "env": env,
            "target_size": target_size
        }

    def _forward(self, model_inputs):
        """
        The Planner identifies the critical structures.
        """
        # The LLM "sees" the graph and suggests groupings
        plan = self.planner.create_plan(
            model_inputs["workflow"], model_inputs["env"], []
        )
        return {
            "logits": plan, 
            "workflow": model_inputs["workflow"],
            "env": model_inputs["env"],
            "target_size": model_inputs["target_size"]
        }

    def postprocess(self, model_outputs, threshold=0.5):
        """
        Convert the plan into 'Bounding Boxes' (Task Clusters).
        """
        # We use the Evaluator to validate the clusters suggested by the plan
        result = self.evaluator.find_best_policy(
             model_outputs["workflow"],
             model_outputs["env"],
             params={},
             plan=model_outputs["logits"]
        )
        
        policy = result["best_policy"]
        if not policy:
            return []
            
        # Group tasks by server (These are our "Boxes")
        clusters = {}
        for task_idx, server_id in enumerate(policy):
            if server_id not in clusters:
                clusters[server_id] = []
            clusters[server_id].append(task_idx + 1)
            
        # Format like Object Detection output
        annotations = []
        for server_id, tasks in clusters.items():
            label = "CLOUD_CLUSTER" if server_id == 1 else "EDGE_CLUSTER"
            
            # Create a "Box" representation (min/max task ID)
            box = {
                "xmin": min(tasks),
                "ymin": server_id, # Server ID acts as Y coordinate
                "xmax": max(tasks),
                "ymax": server_id
            }
            
            annotations.append({
                "score": 1.0 / (result["best_cost"] + 1e-6), # Inverse cost as score
                "label": label,
                "box": box
            })
            
        return annotations