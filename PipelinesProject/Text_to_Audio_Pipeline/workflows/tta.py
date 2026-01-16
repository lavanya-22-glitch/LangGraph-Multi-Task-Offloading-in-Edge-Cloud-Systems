from workflows import Fake, Pack, Unpack
import torch
import scipy.io.wavfile
from transformers import AutoProcessor, MusicgenForConditionalGeneration

def TASK(text_path):
    def NEW(): return Fake(
        pipe = "TTA", 
        stage = None,
        audio_tensor = None, 
        text = None,
        preinputs = None, 
        sampling_rate = None,
    )
    task = NEW()
    task.stage = "TASK"
    with open(text_path, "r") as f: task.text = f.read()
    return task

class PRE:
    def __call__(self, task):
        task.stage = "PRE"
        # MusicGen uses standard text prompts, no special tokens required like Speechless
        task.preinputs = task.text
        del task.text 
        return task

class EXE:
    def __init__(self):
        self.model_id = "facebook/musicgen-melody"
        # Increase timeout to handle slow connections
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, 
            token=None, 
            trust_remote_code=True
        )
        self.model = MusicgenForConditionalGeneration.from_pretrained(
            self.model_id
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def __call__(self, task):
        task.stage = "EXE"

        # Process inputs
        inputs = self.processor(
            text=[task.preinputs],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Generate Audio (max_new_tokens 256 is ~5 seconds of audio)
        audio_values = self.model.generate(**inputs, max_new_tokens=256)
        
        # Store tensor on CPU for pickling/serialization
        task.audio_tensor = audio_values[0, 0].cpu().numpy()
        task.sampling_rate = self.model.config.audio_encoder.sampling_rate
        
        del task.preinputs 
        return task

class POST:
    def __call__(self, task):
        task.stage = "POST"
        
        # Export the audio tensor to a wav file
        output_filename = "output_music.wav"
        scipy.io.wavfile.write(output_filename, rate=task.sampling_rate, data=task.audio_tensor)
        
        task.result = f"Audio saved to {output_filename}"
        
        # Cleanup large tensor data
        del task.audio_tensor
        return task