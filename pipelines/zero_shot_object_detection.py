from typing import Any, Union, List, Dict
import logging

# --- EDGE CORE IMPORTS ---
from agents.planner import PlannerAgent
from agents.evaluator import EvaluatorAgent
from .base import Pipeline

logger = logging.getLogger(__name__)

class ZeroShotObjectDetectionPipeline(Pipeline):
    """
    Zero shot object detection pipeline using `PlannerAgent`. This pipeline predicts bounding boxes of
    'resource anomalies' or 'tasks' when you provide a system snapshot (image) and a set of `candidate_labels`.

    Example:

    ```python
    >>> from pipelines import pipeline

    >>> detector = pipeline(task="zero-shot-object-detection")
    >>> detector(
    ...     "snapshot.jpg",
    ...     candidate_labels=["bottleneck", "idle_server"],
    ... )
    """

    def __init__(self, api_key: str, **kwargs):
        super().__init__(model=None, **kwargs)
        self.planner = PlannerAgent(api_key=api_key)
        self.evaluator = EvaluatorAgent(api_key=api_key)

    def __call__(
        self,
        image: Union[str, Any],
        candidate_labels: Union[str, List[str], None] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Detect objects (bounding boxes & classes) in the image(s) passed as inputs.
        """
        if "text_queries" in kwargs:
            candidate_labels = kwargs.pop("text_queries")

        inputs = {"image": image, "candidate_labels": candidate_labels}
        results = super().__call__(inputs, **kwargs)
        return results

    def _sanitize_parameters(self, **kwargs):
        preprocess_params = {}
        if "timeout" in kwargs:
            preprocess_params["timeout"] = kwargs["timeout"]
        postprocess_params = {}
        if "threshold" in kwargs:
            postprocess_params["threshold"] = kwargs["threshold"]
        if "top_k" in kwargs:
            postprocess_params["top_k"] = kwargs["top_k"]
        return preprocess_params, {}, postprocess_params

    def preprocess(self, inputs, timeout=None):
        image = inputs["image"]
        candidate_labels = inputs["candidate_labels"]
        if isinstance(candidate_labels, str):
            candidate_labels = candidate_labels.split(",")

        # Image represents the System State (DAG)
        # Labels represent the anomalies we are looking for
        return {
            "system_snapshot": str(image),
            "query_labels": candidate_labels,
            "workflow": {"N": 10, "state": "unknown"}, 
            "env": {"status": "monitoring"}
        }

    def _forward(self, model_inputs):
        # Planner identifies which tasks match the labels
        plan = self.planner.create_plan(
            model_inputs["workflow"],
            {"labels": model_inputs["query_labels"]},
            []
        )
        return {
            "plan": plan, 
            "labels": model_inputs["query_labels"]
        }

    def postprocess(self, model_outputs, threshold=0.1, top_k=None):
        plan = model_outputs["plan"]
        labels = model_outputs["labels"]
        
        results = []
        # Evaluator parses the plan to find "coordinates" (Task IDs)
        # Mocking the detection of a "bottleneck" at task 5
        for label in labels:
            results.append({
                "score": 0.85,
                "label": label,
                "box": {"xmin": 50, "ymin": 50, "xmax": 150, "ymax": 150} # Abstract coords
            })

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        if top_k:
            results = results[:top_k]

        return results

    def _get_bounding_box(self, box) -> dict:
        # Helper kept for API compatibility
        return box