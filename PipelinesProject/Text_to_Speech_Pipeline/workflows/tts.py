from . import Fake, Pack, Unpack
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def TASK(text_path):
    def NEW(): return Fake(
        pipe = "TTS", 
        stage = None,
        audio = None, 
        text = None,
        preinputs = None, 
        predictions = None,
    )
    task = NEW()
    task.stage = "TASK"
    with open(text_path, "r") as f: task.text = f.read()
    return task

class PRE:
    def __call__(self, task):
        task.stage = "PRE"
        task.preinputs = f"<|reserved_special_token_69|>{task.text}"
        del task.text 
        return task

class EXE:
    def __init__(self):
        self.model_id = "homebrewltd/Speechless-llama3.2-v0.1"
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            torch_dtype=torch.bfloat16
        )

    def __call__(self, task):
        task.stage = "EXE"

        inputs = self.tokenizer(task.preinputs, return_tensors="pt")
        
        outputs = self.model.generate(**inputs, max_new_tokens=512)
        
        task.predictions = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        del task.preinputs 
        return task

class POST:
    def __call__(self, task):
        task.stage = "POST"
        if "assistant\n\n" in task.predictions:
            task.result = task.predictions.split("assistant\n\n")[-1].strip().upper()
        else:
            task.result = task.predictions
        del task.predictions
        return task