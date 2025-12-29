import logging
from typing import Any, Dict, Tuple, Union

logger = logging.getLogger(__name__)

class Pipeline:
    """
    Base class for all task offloading optimization pipelines.
    Mimics the Hugging Face Pipeline architecture to provide a consistent 
    interface for diverse offloading scenarios.
    """
    def __init__(self, model: Any, device: int = -1, **kwargs):
        """
        Initialize the pipeline.
        
        Args:
            model: The 'Model' object containing your Planner, Evaluator, and Output agents.
            device: Hardware identifier (-1 for CPU, 0+ for GPU).
        """
        self.model = model
        self.device = device
        # Initialize internal configuration from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _sanitize_parameters(self, **kwargs) -> Tuple[Dict, Dict, Dict]:
        """
        Must be implemented by each task pipeline to split kwargs into 
        preprocess, forward, and postprocess buckets.
        """
        raise NotImplementedError("_sanitize_parameters must be implemented by the task pipeline.")

    def preprocess(self, inputs: Any, **kwargs) -> Dict[str, Any]:
        """Convert raw DAG and Environment data into model-ready context."""
        raise NotImplementedError("preprocess must be implemented by the task pipeline.")

    def _forward(self, model_inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute the core Agentic Reasoning (Planner) and Evaluation."""
        raise NotImplementedError("_forward must be implemented by the task pipeline.")

    def postprocess(self, model_outputs: Dict[str, Any], **kwargs) -> Any:
        """Format the final policy and generate the human-readable explanation."""
        raise NotImplementedError("postprocess must be implemented by the task pipeline.")

    def __call__(self, inputs: Any, **kwargs) -> Any:
        """
        The main entry point. Orchestrates the pipeline flow.
        Identical to the call structure in any_to_any.py and mask_generation.py.
        """
        # 1. Sanitize the parameters for each stage
        preprocess_params, forward_params, postprocess_params = self._sanitize_parameters(**kwargs)

        # 2. Sequential execution
        model_inputs = self.preprocess(inputs, **preprocess_params)
        model_outputs = self._forward(model_inputs, **forward_params)
        outputs = self.postprocess(model_outputs, **postprocess_params)

        return outputs