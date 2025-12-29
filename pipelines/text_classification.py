import inspect
import warnings
from typing import Any, Dict, List, Union

import numpy as np

# --- EDGE COMPUTING CORE IMPORTS ---
from core.workflow import Workflow
from core.environment import Environment
from core.memory_manager import WorkflowMemory
from agents.planner import PlannerAgent
from agents.evaluator import EvaluatorAgent

# Mocking internal utilities to match reference structure
class ExplicitEnum(str, Enum):
    @classmethod
    def _missing_(cls, value):
        return cls(value)

logger = logging.getLogger(__name__)

def sigmoid(_outputs):
    return 1.0 / (1.0 + np.exp(-_outputs))

def softmax(_outputs):
    maxes = np.max(_outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(_outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

class ClassificationFunction(ExplicitEnum):
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    NONE = "none"

class TextClassificationPipeline:
    """
    Text classification pipeline using `PlannerAgent` logic (acting as ModelForSequenceClassification). 
    See the [sequence classification examples](../task_summary#sequence-classification) for more information.

    This pipeline predicts optimal task placement labels (CLOUD, EDGE, LOCAL) for a given workflow task.

    Example:

    ```python
    >>> from pipelines import pipeline

    >>> classifier = pipeline(task="task-classification", api_key="sk-...")
    >>> classifier(workflow_data)
    [{'label': 'CLOUD', 'score': 1.0}]
    ```

    If multiple classification labels are available, the pipeline will run a softmax over the results. 
    """

    _load_processor = False
    _load_image_processor = False
    _load_feature_extractor = False
    _load_tokenizer = True

    return_all_scores = False
    function_to_apply = ClassificationFunction.NONE

    def __init__(self, api_key: str, memory_dir: str = "memory_store", **kwargs):
        """
        Initialize the pipeline with Agentic Logic.
        """
        self.memory_manager = WorkflowMemory(memory_dir=memory_dir)
        self.planner = PlannerAgent(api_key=api_key)
        self.evaluator = EvaluatorAgent(api_key=api_key)
        
        # Mocking model config for the pipeline structure
        self.model = type('MockModel', (), {'config': type('Config', (), {'id2label': {0: 'LOCAL', 1: 'CLOUD'}, 'num_labels': 2})})

    def _sanitize_parameters(self, return_all_scores=None, function_to_apply=None, top_k="", **kwargs):
        preprocess_params = kwargs
        postprocess_params = {}

        if isinstance(top_k, int) or top_k is None:
            postprocess_params["top_k"] = top_k
            postprocess_params["_legacy"] = False
        
        if isinstance(function_to_apply, str):
            function_to_apply = ClassificationFunction[function_to_apply.upper()]

        if function_to_apply is not None:
            postprocess_params["function_to_apply"] = function_to_apply
            
        return preprocess_params, {}, postprocess_params

    def __call__(
        self,
        inputs: Union[Dict, List[Dict]],
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Classify the workflow tasks given as inputs.
        """
        # Mimic strict input handling
        if not isinstance(inputs, (dict, list)):
             raise ValueError("Inputs must be a dictionary (workflow) or list of dictionaries.")

        preprocess_params, forward_params, postprocess_params = self._sanitize_parameters(**kwargs)
        
        # 1. Preprocess
        model_inputs = self.preprocess(inputs, **preprocess_params)
        
        # 2. Forward
        model_outputs = self._forward(model_inputs)
        
        # 3. Postprocess
        result = self.postprocess(model_outputs, **postprocess_params)
        
        return result

    def preprocess(self, inputs, **kwargs) -> Dict[str, Any]:
        """
        Normalize inputs and retrieve memory context.
        """
        # Support both single dict and list format
        if isinstance(inputs, list):
            inputs = inputs[0]
            
        workflow = inputs.get("workflow")
        env = inputs.get("env")
        
        # Context Retrieval (Replacing Tokenization)
        workflow_obj = Workflow.from_experiment_dict(workflow)
        examples = self.memory_manager.retrieve_similar_executions(workflow_obj, top_k=3)
        
        return {
            "workflow": workflow, 
            "env": env, 
            "examples": examples,
            "original_obj": workflow_obj
        }

    def _forward(self, model_inputs):
        """
        The Planner generates the classification plan.
        """
        # The Planner acts as the Classifier here
        plan_text = self.planner.create_plan(
            model_inputs["workflow"], 
            model_inputs["env"], 
            model_inputs["examples"]
        )
        
        # Return in a format similar to ModelOutput
        return {
            "logits": plan_text, 
            "model_inputs": model_inputs
        }

    def postprocess(self, model_outputs, function_to_apply=None, top_k=1, _legacy=True):
        """
        Convert LLM plan into labels and scores.
        """
        plan = model_outputs["logits"]
        inputs = model_outputs["model_inputs"]
        
        # Use Evaluator to turn "Logits" (Text) into "Probabilities" (Cost Scores)
        result = self.evaluator.find_best_policy(
             inputs["workflow"], inputs["env"], params={}, plan=plan
        )
        
        best_policy = result["best_policy"]
        
        # Format exactly like TextClassification output
        dict_scores = []
        if best_policy:
            for i, loc_id in enumerate(best_policy):
                label = "LOCAL" if loc_id == 0 else "CLOUD" if loc_id == 1 else f"EDGE_{loc_id}"
                dict_scores.append({
                    "label": label,
                    "score": 1.0, # Deterministic confidence
                    "task_id": i+1
                })
                
        return dict_scores