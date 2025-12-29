from typing import Any, Dict, List, Optional, Union
import logging

# --- EDGE COMPUTING CORE IMPORTS ---
from core.workflow import Workflow
from core.memory_manager import WorkflowMemory
from agents.planner import PlannerAgent
from agents.evaluator import EvaluatorAgent

logger = logging.getLogger(__name__)

class MaskGenerationPipeline:
    """
    Automatic mask generation pipeline using `PlannerAgent`. 
    This pipeline predicts optimal task placement vectors (masks) for a generic Directed Acyclic Graph (DAG).

    This pipeline works in 3 specific steps (Preprocess -> Forward -> Postprocess) to match
    standard generation architectures.

    Example:
    ```python
    >>> from pipelines import pipeline
    >>> masker = pipeline(task="mask-generation")
    >>> masker(workflow_data)
    {'masks': [0, 1, 1, 0], 'scores': 20.5}
    ```
    """

    _load_processor = False
    _load_image_processor = False
    _load_feature_extractor = False
    _load_tokenizer = None

    def __init__(self, api_key: str, memory_dir: str = "memory_store", **kwargs):
        """
        Initialize the pipeline components (Agents and Memory).
        """
        self.memory_manager = WorkflowMemory(memory_dir=memory_dir)
        self.planner = PlannerAgent(api_key=api_key)
        self.evaluator = EvaluatorAgent(api_key=api_key)
        self._sanitize_parameters(**kwargs)

    def _sanitize_parameters(self, **kwargs):
        """
        Splits kwargs into preprocessing, forwarding, and postprocessing arguments.
        """
        preprocess_kwargs = {}
        forward_params = {}
        postprocess_kwargs = {}

        if "retrieval_k" in kwargs:
            preprocess_kwargs["retrieval_k"] = kwargs["retrieval_k"]
        if "points_per_batch" in kwargs:
             preprocess_kwargs["retrieval_k"] = kwargs["points_per_batch"]

        if "temperature" in kwargs:
            forward_params["temperature"] = kwargs["temperature"]
        if "model_version" in kwargs:
            forward_params["model_version"] = kwargs["model_version"]

        if "max_latency" in kwargs:
            postprocess_kwargs["max_latency"] = kwargs["max_latency"]
        if "pred_iou_thresh" in kwargs:
             postprocess_kwargs["max_latency"] = kwargs["pred_iou_thresh"]

        self.preprocess_kwargs = preprocess_kwargs
        self.forward_params = forward_params
        self.postprocess_kwargs = postprocess_kwargs
        
        return preprocess_kwargs, forward_params, postprocess_kwargs

    def __call__(
        self, 
        workflow_data: Union[Dict, List[Dict]], 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generates optimal offloading policies (masks).
        """
        # 1. Parameter Sanitation
        self._sanitize_parameters(**kwargs)

        # 2. Preprocess (Context Building)
        model_inputs = self.preprocess(workflow_data, **self.preprocess_kwargs)

        # 3. Forward (Inference)
        model_outputs = self._forward(model_inputs, **self.forward_params)

        # 4. Postprocess (Filtering & Cost Calculation)
        final_output = self.postprocess(model_outputs, **self.postprocess_kwargs)

        return final_output

    def preprocess(
        self,
        workflow_data: Dict,
        retrieval_k: int = 3,
        timeout: float = None,
    ) -> Dict[str, Any]:
        """
        Prepares the input for the model. 
        Retrieves similar past executions (embeddings) to form the Few-Shot Context.
        """
        workflow_dict = workflow_data.get("workflow")
        env_dict = workflow_data.get("env")

        # Feature Extraction (Embedding)
        workflow_obj = Workflow.from_experiment_dict(workflow_dict)
        
        # Memory Retrieval
        retrieved_examples = self.memory_manager.retrieve_similar_executions(
            workflow_obj, top_k=retrieval_k
        )

        return {
            "workflow": workflow_dict,
            "env": env_dict,
            "examples": retrieved_examples,
            "original_obj": workflow_obj
        }

    def _forward(
        self,
        model_inputs: Dict[str, Any],
        temperature: float = 0.7,
        model_version: str = "gpt-4"
    ) -> Dict[str, Any]:
        """
        Feeds the outputs of `preprocess` to the Model (LLM).
        """
        workflow = model_inputs["workflow"]
        env = model_inputs["env"]
        examples = model_inputs["examples"]

        # The "Model" Forward Pass
        raw_plan_text = self.planner.create_plan(
            workflow, env, examples
        )

        return {
            "plan_text": raw_plan_text,
            "workflow": workflow,
            "env": env,
            "original_obj": model_inputs["original_obj"]
        }

    def postprocess(
        self,
        model_outputs: Dict[str, Any],
        max_latency: float = None,
    ) -> Dict[str, Any]:
        """
        Filters masks (policies) based on Cost and Constraints.
        """
        plan = model_outputs["plan_text"]
        workflow = model_outputs["workflow"]
        env = model_outputs["env"]
        
        result = self.evaluator.find_best_policy(
            workflow, env, params={"max_latency": max_latency}, plan=plan
        )

        best_policy = result.get("best_policy")
        best_cost = result.get("best_cost")

        if best_policy:
            self.memory_manager.save_execution(
                model_outputs["original_obj"],
                best_policy,
                best_cost
            )

        return {
            "masks": best_policy,
            "scores": best_cost,
            "reasoning": plan 
        }