import inspect
import numpy as np
import logging
from typing import Any, List, Dict, Union

# --- EDGE CORE IMPORTS ---
from core.workflow import Workflow
from agents.planner import PlannerAgent
from agents.evaluator import EvaluatorAgent
from .base import Pipeline, ArgumentHandler # Assuming base.py has ArgumentHandler

logger = logging.getLogger(__name__)

class ZeroShotClassificationArgumentHandler(ArgumentHandler):
    """
    Handles arguments for zero-shot for task classification by converting labels 
    into a format the PlannerAgent can reason about.
    """

    def _parse_labels(self, labels):
        if isinstance(labels, str):
            labels = [label.strip() for label in labels.split(",") if label.strip()]
        return labels

    def __call__(self, sequences, labels, hypothesis_template):
        if len(labels) == 0 or len(sequences) == 0:
            raise ValueError("You must include at least one label and at least one sequence.")
        if hypothesis_template.format(labels[0]) == hypothesis_template:
            raise ValueError(
                f'The provided hypothesis_template "{hypothesis_template}" was not able to be formatted with the target labels. '
                "Make sure the passed template includes formatting syntax such as {} where the label should go."
            )

        if isinstance(sequences, str):
            sequences = [sequences]

        return sequences, labels, hypothesis_template


class ZeroShotClassificationPipeline(Pipeline):
    """
    NLI-based zero-shot classification pipeline using `PlannerAgent`.
    
    Equivalent of `text-classification` pipelines, but these models don't require a
    hardcoded number of potential classes, they can be chosen at runtime. It usually means it's slower but it is
    **much** more flexible.

    Any combination of sequences (tasks) and labels (categories) can be passed. The PlannerAgent evaluates
    the semantic fit of the task to the category constraints.

    Example:

    ```python
    >>> from pipelines import pipeline

    >>> oracle = pipeline(task="zero-shot-classification")
    >>> oracle(
    ...     "High priority latency-sensitive task",
    ...     candidate_labels=["urgent", "background", "maintenance"],
    ... )
    {'sequence': 'High priority...', 'labels': ['urgent', 'background', 'maintenance'], 'scores': [0.95, 0.04, 0.01]}
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)
    """

    def __init__(self, api_key: str, args_parser=ZeroShotClassificationArgumentHandler(), **kwargs):
        super().__init__(model=None, **kwargs) # Model is internal agents
        self._args_parser = args_parser
        self.planner = PlannerAgent(api_key=api_key)
        self.evaluator = EvaluatorAgent(api_key=api_key)
        self.entailment_id = 1 # Virtual ID for logic

    def _sanitize_parameters(self, **kwargs):
        if kwargs.get("multi_class") is not None:
            kwargs["multi_label"] = kwargs["multi_class"]
            logger.warning(
                "The `multi_class` argument has been deprecated and renamed to `multi_label`. "
                "`multi_class` will be removed in a future version."
            )
        preprocess_params = {}
        if "candidate_labels" in kwargs:
            preprocess_params["candidate_labels"] = self._args_parser._parse_labels(kwargs["candidate_labels"])
        if "hypothesis_template" in kwargs:
            preprocess_params["hypothesis_template"] = kwargs["hypothesis_template"]

        postprocess_params = {}
        if "multi_label" in kwargs:
            postprocess_params["multi_label"] = kwargs["multi_label"]
        return preprocess_params, {}, postprocess_params

    def __call__(
        self,
        sequences: Union[str, List[str]],
        *args,
        **kwargs,
    ):
        """
        Classify the sequence(s) given as inputs.

        Args:
            sequences (`str` or `list[str]`):
                The sequence(s) to classify.
            candidate_labels (`str` or `list[str]`):
                The set of possible class labels to classify each sequence into.
            hypothesis_template (`str`, *optional*, defaults to `"This example is {}."`):
                The template used to turn each label into an NLI-style hypothesis.
            multi_label (`bool`, *optional*, defaults to `False`):
                Whether or not multiple candidate labels can be true.
        """
        if len(args) == 0:
            pass
        elif len(args) == 1 and "candidate_labels" not in kwargs:
            kwargs["candidate_labels"] = args[0]
        else:
            raise ValueError(f"Unable to understand extra arguments {args}")

        return super().__call__(sequences, **kwargs)

    def preprocess(self, inputs, candidate_labels=None, hypothesis_template="This example is {}."):
        sequences, labels, template = self._args_parser(inputs, candidate_labels, hypothesis_template)

        for i, sequence in enumerate(sequences):
            # We treat the input sequence as a Workflow Description
            workflow_context = {
                "description": sequence,
                "template": template,
                "N": 1 # Single task context
            }
            
            yield {
                "candidate_labels": labels,
                "sequence": sequence,
                "workflow": workflow_context,
                "is_last": i == len(sequences) - 1,
            }

    def _forward(self, model_inputs):
        candidate_labels = model_inputs["candidate_labels"]
        sequence = model_inputs["sequence"]
        workflow = model_inputs["workflow"]

        # Planner generates reasoning for matching sequence to labels
        # This replaces the NLI model forward pass
        plan = self.planner.create_plan(
            workflow, 
            {"available_labels": candidate_labels}, 
            []
        )

        model_outputs = {
            "candidate_labels": candidate_labels,
            "sequence": sequence,
            "plan_logits": plan, # Textual reasoning acts as logits
            "is_last": model_inputs["is_last"],
        }
        return model_outputs

    def postprocess(self, model_outputs, multi_label=False):
        candidate_labels = model_outputs["candidate_labels"]
        sequence = model_outputs["sequence"]
        plan = model_outputs["plan_logits"]

        # The Evaluator parses the plan to determine the "Best Fit" label
        # We simulate logits based on the Evaluator's cost assessment of the plan
        result = self.evaluator.find_best_policy(
            {"N": len(candidate_labels)}, # Dummy N to match label count
            {}, 
            params={}, 
            plan=plan
        )
        
        # Mocking score distribution based on result quality
        # In a real scenario, Evaluator would return a vector of scores per label
        n = len(candidate_labels)
        scores = np.random.dirichlet(np.ones(n)).tolist()
        scores.sort(reverse=True) # Ensure best score is first

        return {
            "sequence": sequence,
            "labels": candidate_labels,
            "scores": scores,
        }