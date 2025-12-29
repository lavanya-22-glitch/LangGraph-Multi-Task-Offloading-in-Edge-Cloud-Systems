import enum
from typing import Any, List, Dict, Union
import logging

# --- EDGE CORE IMPORTS ---
from agents.planner import PlannerAgent
from agents.evaluator import EvaluatorAgent
from .base import Pipeline

logger = logging.getLogger(__name__)

class ReturnType(enum.Enum):
    TENSORS = 0
    NEW_TEXT = 1
    FULL_TEXT = 2

class TextGenerationPipeline(Pipeline):
    """
    Language generation pipeline using `PlannerAgent`.
    Generates text (Configs, Scripts, Explanations) following a prompt.
    """

    def __init__(self, api_key: str, **kwargs):
        super().__init__(model=None, **kwargs)
        self.planner = PlannerAgent(api_key=api_key)
        self.evaluator = EvaluatorAgent(api_key=api_key)
        # Mock generation config
        self.generation_config = type('GC', (), {'max_length': 256, 'max_new_tokens': 256})

    def _sanitize_parameters(
        self,
        return_full_text=None,
        return_tensors=None,
        return_text=None,
        return_type=None,
        max_length=None,
        **generate_kwargs,
    ):
        preprocess_params = {}
        if max_length is not None:
            preprocess_params["max_length"] = max_length

        postprocess_params = {}
        if return_full_text is not None and return_type is None:
            return_type = ReturnType.FULL_TEXT if return_full_text else ReturnType.NEW_TEXT
        if return_type is not None:
            postprocess_params["return_type"] = return_type

        return preprocess_params, generate_kwargs, postprocess_params

    def __call__(self, text_inputs, **kwargs):
        """
        Complete the prompt(s) given as inputs.
        """
        return super().__call__(text_inputs, **kwargs)

    def preprocess(
        self,
        prompt_text,
        max_length=None,
        **generate_kwargs,
    ):
        # Prepare context for the Planner
        return {
            "prompt": prompt_text,
            "workflow": {"description": prompt_text, "N": 1},
            "env": {},
            "generate_args": generate_kwargs
        }

    def _forward(self, model_inputs, **generate_kwargs):
        # Planner generates the content (The "LLM" step)
        text = self.planner.create_plan(
            model_inputs["workflow"],
            model_inputs["env"],
            []
        )
        
        return {
            "generated_sequence": text,
            "prompt_text": model_inputs["prompt"]
        }

    def postprocess(
        self,
        model_outputs,
        return_type=ReturnType.FULL_TEXT,
    ):
        generated_text = model_outputs["generated_sequence"]
        prompt = model_outputs["prompt_text"]
        
        if return_type == ReturnType.FULL_TEXT:
            final_text = prompt + "\n" + generated_text
        else:
            final_text = generated_text
            
        return [{"generated_text": final_text}]