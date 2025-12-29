from typing import Union, List, Dict, Any
import logging

# --- EDGE CORE IMPORTS ---
from agents.planner import PlannerAgent
from agents.evaluator import EvaluatorAgent
from .base import Pipeline

logger = logging.getLogger(__name__)

class VisualQuestionAnsweringPipeline(Pipeline):
    """
    Visual Question Answering pipeline using `PlannerAgent`. 
    Answers questions about system dashboards (images).
    """

    def __init__(self, api_key: str, **kwargs):
        super().__init__(model=None, **kwargs)
        self.planner = PlannerAgent(api_key=api_key)
        self.evaluator = EvaluatorAgent(api_key=api_key)
        self.model = type('Mock', (), {'config': type('Cfg', (), {'id2label': {0: 'yes', 1: 'no'}, 'num_labels': 2})})

    def _sanitize_parameters(self, top_k=None, padding=None, truncation=None, timeout=None, **kwargs):
        postprocess_params = {}
        if top_k is not None:
            postprocess_params["top_k"] = top_k
        return {}, {}, postprocess_params

    def __call__(
        self,
        image: Union[str, Any],
        question: Union[str, List[str], None] = None,
        **kwargs,
    ):
        """
        Answers open-ended questions about images.
        """
        if isinstance(image, str) and isinstance(question, str):
            inputs = {"image": image, "question": question}
        else:
            inputs = image # Assumption: formatted dict
            
        results = super().__call__(inputs, **kwargs)
        return results

    def preprocess(self, inputs, padding=False, truncation=False, timeout=None):
        # Prepare context
        return {
            "question": inputs.get("question"),
            "image_ref": str(inputs.get("image")),
            "workflow": {"N": 1, "context": "dashboard_analysis"},
            "env": {"visual_data": "loaded"}
        }

    def _forward(self, model_inputs, **generate_kwargs):
        # Planner answers the question based on the 'image' context
        answer = self.planner.create_plan(
            model_inputs["workflow"],
            {"question": model_inputs["question"], "image": model_inputs["image_ref"]},
            []
        )
        return {"answer": answer}

    def postprocess(self, model_outputs, top_k=5):
        # Format as VQA answer
        return [{"score": 0.99, "answer": model_outputs["answer"]}]