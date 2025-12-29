from typing import Any, TypedDict, List, Union
import logging
import numpy as np

# --- EDGE CORE IMPORTS ---
from agents.planner import PlannerAgent
from agents.evaluator import EvaluatorAgent
from .base import Pipeline

DEFAULT_VOCODER_ID = "microsoft/speecht5_hifigan"

class AudioOutput(TypedDict, total=False):
    """
    audio (`np.ndarray`):
        The generated audio waveform.
    sampling_rate (`int`):
        The sampling rate of the generated audio waveform.
    """
    audio: Any
    sampling_rate: int

class TextToAudioPipeline(Pipeline):
    """
    Text-to-audio generation pipeline.
    Simulates converting Edge Alerts (Text) to Audio signals/logs.
    """

    def __init__(self, api_key: str, vocoder=None, sampling_rate=None, **kwargs):
        super().__init__(model=None, **kwargs)
        self.planner = PlannerAgent(api_key=api_key)
        self.evaluator = EvaluatorAgent(api_key=api_key)
        self.sampling_rate = sampling_rate if sampling_rate else 16000

    def preprocess(self, text, **kwargs):
        if isinstance(text, str):
            text = [text]
        
        # Prepare text as a Task Description
        return {
            "text_input": text,
            "workflow": {"N": 1, "task": "alert_generation"}, 
            "env": {}
        }

    def _forward(self, model_inputs, **kwargs):
        # Planner decides the "Tone" of the audio based on text urgency
        plan = self.planner.create_plan(
            model_inputs["workflow"],
            {"content": model_inputs["text_input"]},
            []
        )
        return {"plan": plan}

    def __call__(self, text_inputs, **forward_params):
        """
        Generates speech/audio from the inputs.
        """
        return super().__call__(text_inputs, **forward_params)

    def _sanitize_parameters(self, preprocess_params=None, forward_params=None, generate_kwargs=None):
        return {}, {}, {}

    def postprocess(self, model_outputs):
        # Return mock audio data representing the alert
        # Shape: (1, 1000) dummy waveform
        audio_data = np.random.uniform(-1, 1, 16000) 
        
        return AudioOutput(
            audio=audio_data,
            sampling_rate=self.sampling_rate,
        )