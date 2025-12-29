import types
import warnings
from typing import Any, List, Dict, Union, Tuple
import numpy as np
import logging

# --- EDGE CORE IMPORTS ---
from agents.planner import PlannerAgent
from agents.evaluator import EvaluatorAgent
from .base import Pipeline, ArgumentHandler

class AggregationStrategy:
    NONE = "none"
    SIMPLE = "simple"
    FIRST = "first"
    AVERAGE = "average"
    MAX = "max"

class TokenClassificationArgumentHandler(ArgumentHandler):
    """
    Handles arguments for token classification.
    """
    def __call__(self, inputs: Union[str, List[str]], **kwargs):
        is_split_into_words = kwargs.get("is_split_into_words", False)
        delimiter = kwargs.get("delimiter")
        if isinstance(inputs, str):
            inputs = [inputs]
        return inputs, is_split_into_words, None, delimiter

class TokenClassificationPipeline(Pipeline):
    """
    Named Entity Recognition pipeline using `PlannerAgent`. 
    Extracts entities (Server IDs, IPs, Error Codes) from logs.
    """

    def __init__(self, api_key: str, args_parser=TokenClassificationArgumentHandler(), **kwargs):
        super().__init__(model=None, **kwargs)
        self.planner = PlannerAgent(api_key=api_key)
        self.evaluator = EvaluatorAgent(api_key=api_key)
        self._args_parser = args_parser
        self.tokenizer = type('MockTok', (), {'is_fast': False, 'model_max_length': 512})

    def _sanitize_parameters(
        self,
        ignore_labels=None,
        aggregation_strategy=None,
        **kwargs
    ):
        preprocess_params = {}
        postprocess_params = {}
        if aggregation_strategy is not None:
            postprocess_params["aggregation_strategy"] = aggregation_strategy
        if ignore_labels is not None:
            postprocess_params["ignore_labels"] = ignore_labels
        return preprocess_params, {}, postprocess_params

    def __call__(self, inputs: Union[str, List[str]], **kwargs):
        """
        Classify each token of the text(s) given as inputs.
        """
        _inputs, is_split_into_words, offset_mapping, delimiter = self._args_parser(inputs, **kwargs)
        kwargs["is_split_into_words"] = is_split_into_words
        
        return super().__call__(inputs, **kwargs)

    def preprocess(self, sentence, **kwargs):
        # Treat log line as a task description
        if isinstance(sentence, list):
            sentence = sentence[0]
            
        return {
            "sentence": sentence,
            "workflow": {"N": 1, "task": "log_entity_extraction"},
            "env": {}
        }

    def _forward(self, model_inputs):
        # Planner identifies entities in text
        plan = self.planner.create_plan(
            model_inputs["workflow"],
            {"log_text": model_inputs["sentence"]},
            []
        )
        return {"plan": plan, "sentence": model_inputs["sentence"]}

    def postprocess(self, model_outputs, aggregation_strategy=None, ignore_labels=None):
        # Mocking entity extraction from plan
        # In a real impl, you'd parse the 'plan' string for entities
        return [{
            "entity_group": "SERVER_ID",
            "score": 0.99,
            "word": "Server-01",
            "start": 0,
            "end": 9
        }]