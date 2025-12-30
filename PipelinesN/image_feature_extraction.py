from typing import Any, Union, Dict, List
import numpy as np
import torch
from .base import Pipeline

# Core Project Dependencies
from core.workflow import Workflow
from core.memory_manager import WorkflowMemory

class OffloadingImageFeatureExtractionPipeline(Pipeline):
    """
    Feature extraction pipeline for offloading scenarios.
    Extracts numerical 'hidden states' representing the structural 
    characteristics of a DAG and its environment.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The WorkflowMemory acts as the 'Image Processor' for offloading data
        self.memory_processor = WorkflowMemory()

    def _sanitize_parameters(self, pool=None, return_tensors=None, **kwargs):
        """Standardized parameter sanitizer."""
        preprocess_params = {}
        postprocess_params = {}
        
        if pool is not None:
            postprocess_params["pool"] = pool
        if return_tensors is not None:
            postprocess_params["return_tensors"] = return_tensors
        if "timeout" in kwargs:
            preprocess_params["timeout"] = kwargs["timeout"]

        return preprocess_params, {}, postprocess_params

    def preprocess(self, scenario_data: Dict, **kwargs) -> Dict:
        """
        Equivalent to image loading. Converts raw input into feature components.
        """
        workflow_dict = scenario_data.get("workflow", {})
        env_dict = scenario_data.get("env", {})
        
        # Extract features using our core memory logic
        wf_features = self.memory_processor.extract_workflow_features(workflow_dict)
        env_features = self.memory_processor.extract_environment_features(env_dict)
        
        return {
            "wf_features": wf_features,
            "env_features": env_features,
            "params": scenario_data.get("costs", {})
        }

    def _forward(self, model_inputs: Dict) -> Dict:
        """
        Equivalent to the model forward pass. Computes the normalized feature vector.
        """
        # Combine structural and physical features into a single vector
        # This is our 'hidden state' representation
        feature_vector = self.memory_processor.compute_feature_vector(
            model_inputs["wf_features"], 
            model_inputs["env_features"]
        )
        
        return {
            "last_hidden_state": feature_vector,
            "pooler_output": np.mean(feature_vector) # Example 'pooled' representation
        }

    def postprocess(self, model_outputs: Dict, pool: bool = False, return_tensors: bool = False):
        """
        Returns the raw features or the pooled summary.
        """
        if pool:
            # Return the single 'regime' value or summary scalar
            outputs = model_outputs["pooler_output"]
        else:
            # Return the full multi-dimensional feature vector
            outputs = model_outputs["last_hidden_state"]

        if return_tensors:
            return torch.tensor(outputs)
        return outputs.tolist()

    def __call__(self, scenario: Dict, **kwargs) -> List[Any]:
        """
        Extract the features of the input offloading scenario.
        """
        return super().__call__(scenario, **kwargs)