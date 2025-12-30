import numpy as np
import torch
import logging
from typing import Any, Dict, List, Union
from .base import Pipeline, build_pipeline_init_args

# Core Dependencies
from core.workflow import Workflow
from core.environment import Environment

logger = logging.getLogger(__name__)

class OffloadingScenarioClassificationPipeline(Pipeline):
    """
    Pipeline for classifying offloading scenarios into performance regimes.
    
    Analogy to ImageClassificationPipeline:
    - Image Input -> Scenario (DAG + Environment)
    - Class Labels -> Regimes (Network-Bound, Compute-Bound, etc.)
    - Softmax Score -> Confidence in regime detection
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Mapping for human-readable labels
        self.id2label = {0: "network-bound", 1: "compute-bound", 2: "energy-bound", 3: "balanced"}

    def _sanitize_parameters(self, top_k=None, **kwargs):
        """Standardized parameter sanitizer for classification."""
        preprocess_params = {}
        postprocess_params = {"top_k": top_k if top_k is not None else 3}
        return preprocess_params, {}, postprocess_params

    def preprocess(self, inputs: Dict, **kwargs):
        """
        Extracts environmental features to determine the regime.
        """
        workflow_obj = Workflow.from_experiment_dict(inputs.get("workflow", {}))
        env_dict = inputs.get("env", {})
        
        # Extract features for classification logic
        dr_mean = np.mean([v for v in env_dict.get('DR', {}).values() if v > 0])
        vr_mean = np.mean(list(env_dict.get('VR', {}).values()))
        ve_mean = np.mean(list(env_dict.get('VE', {}).values()))
        
        return {
            "dr": dr_mean, 
            "vr": vr_mean, 
            "ve": ve_mean, 
            "workflow": workflow_obj
        }

    def _forward(self, model_inputs: Dict):
        """
        Computes the 'logits' (raw scores) for each regime based on physics 
        thresholds.
        """
        # Heuristic-based classification scores (analogy to model logits)
        #
        logits = torch.tensor([
            model_inputs["dr"] * 1e5,   # Network score
            model_inputs["vr"] * 1e8,   # Compute score
            model_inputs["ve"] * 2e6,   # Energy score
            1.0                          # Balanced baseline
        ])
        
        return {"logits": logits.unsqueeze(0)}

    def postprocess(self, model_outputs: Dict, top_k=3):
        """
        Applies softmax to logits and returns sorted regime labels 
       .
        """
        logits = model_outputs["logits"][0]
        # Standard softmax application
        scores = torch.nn.functional.softmax(logits, dim=-1)
        
        dict_scores = [
            {"label": self.id2label[i], "score": score.item()} 
            for i, score in enumerate(scores)
        ]
        
        # Sort by score descending
        dict_scores.sort(key=lambda x: x["score"], reverse=True)
        return dict_scores[:top_k]