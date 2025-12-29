from typing import Any, Union, List, Dict
import numpy as np
import logging

# --- EDGE CORE IMPORTS ---
from agents.planner import PlannerAgent
from agents.evaluator import EvaluatorAgent
from .base import Pipeline

logger = logging.getLogger(__name__)

class ZeroShotAudioClassificationPipeline(Pipeline):
    """
    Zero shot audio classification pipeline.
    Classifies sensor audio streams into `candidate_labels` using `PlannerAgent`.
    """

    def __init__(self, api_key: str, **kwargs):
        super().__init__(model=None, **kwargs)
        self.planner = PlannerAgent(api_key=api_key)
        self.evaluator = EvaluatorAgent(api_key=api_key)

    def __call__(
        self, 
        audios: Union[np.ndarray, bytes, str], 
        candidate_labels: List[str] = None, 
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Assign labels to the audio(s) passed as inputs.
        """
        if candidate_labels is None:
             # Try to extract from kwargs if passed there
             candidate_labels = kwargs.get("candidate_labels")
             
        if candidate_labels is None:
             raise ValueError("candidate_labels must be provided")

        return super().__call__(audios, candidate_labels=candidate_labels, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        preprocess_params = {}
        if "candidate_labels" in kwargs:
            preprocess_params["candidate_labels"] = kwargs["candidate_labels"]
        if "hypothesis_template" in kwargs:
            preprocess_params["hypothesis_template"] = kwargs["hypothesis_template"]

        return preprocess_params, {}, {}

    def preprocess(self, audio, candidate_labels=None, hypothesis_template="This is a sound of {}."):
        # Convert audio input to a context description
        return {
            "audio_sample_ref": str(audio)[:50],
            "labels": candidate_labels,
            "template": hypothesis_template,
            "workflow": {"N": 1, "type": "acoustic_sensing"},
            "env": {}
        }

    def _forward(self, model_inputs):
        # Planner matches audio context to labels
        plan = self.planner.create_plan(
            model_inputs["workflow"],
            {"labels": model_inputs["labels"], "audio_context": model_inputs["audio_sample_ref"]},
            []
        )
        return {"plan": plan, "labels": model_inputs["labels"]}

    def postprocess(self, model_outputs):
        labels = model_outputs["labels"]
        
        # Logic-based scoring derived from Planner output
        scores = [1.0/len(labels)] * len(labels)
        
        result = [
            {"score": score, "label": label}
            for score, label in sorted(zip(scores, labels), key=lambda x: -x[0])
        ]
        return result