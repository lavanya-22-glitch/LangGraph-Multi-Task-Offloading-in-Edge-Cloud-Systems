import warnings
from typing import Any, Union, List, Dict
import logging

# --- EDGE CORE IMPORTS ---
from agents.planner import PlannerAgent
from agents.evaluator import EvaluatorAgent
from .base import Pipeline

logger = logging.getLogger(__name__)

class ZeroShotImageClassificationPipeline(Pipeline):
    """
    Zero shot image classification pipeline using `PlannerAgent`. 
    Predicts the class of a system snapshot (image) given `candidate_labels`.
    """

    def __init__(self, api_key: str, **kwargs):
        super().__init__(model=None, **kwargs)
        self.planner = PlannerAgent(api_key=api_key)
        self.evaluator = EvaluatorAgent(api_key=api_key)

    def __call__(
        self,
        image: Union[str, Any],
        candidate_labels: List[str],
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Assign labels to the image(s) passed as inputs.
        """
        if "images" in kwargs:
            image = kwargs.pop("images")
        if image is None:
            raise ValueError("Cannot call the zero-shot-image-classification pipeline without an images argument!")
        return super().__call__(image, candidate_labels=candidate_labels, **kwargs)

    def _sanitize_parameters(self, tokenizer_kwargs=None, **kwargs):
        preprocess_params = {}
        if "candidate_labels" in kwargs:
            preprocess_params["candidate_labels"] = kwargs["candidate_labels"]
        if "timeout" in kwargs:
            preprocess_params["timeout"] = kwargs["timeout"]
        return preprocess_params, {}, {}

    def preprocess(
        self,
        image,
        candidate_labels=None,
        hypothesis_template="This is a photo of {}.",
        timeout=None,
    ):
        # Convert image reference to System Context
        return {
            "image_ref": str(image),
            "candidate_labels": candidate_labels,
            "workflow": {"N": 5, "type": "system_diagram"},
            "env": {}
        }

    def _forward(self, model_inputs):
        candidate_labels = model_inputs["candidate_labels"]
        
        # Planner analyzes the "Diagram" (System State) against labels
        plan = self.planner.create_plan(
            model_inputs["workflow"],
            {"labels": candidate_labels},
            []
        )
        
        return {
            "plan": plan,
            "candidate_labels": candidate_labels
        }

    def postprocess(self, model_outputs):
        candidate_labels = model_outputs["candidate_labels"]
        
        # Dummy scoring based on label position
        scores = [1.0 / (i + 1) for i in range(len(candidate_labels))]
        total = sum(scores)
        scores = [s / total for s in scores]

        result = [
            {"score": score, "label": candidate_label}
            for score, candidate_label in sorted(zip(scores, candidate_labels), key=lambda x: -x[0])
        ]
        return result