from workflows import Fake, Pack, Unpack
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def TASK(text_path):
    def NEW(): return Fake(
        pipe = "TRANSLATION", 
        stage = None,
        text = None, 
        preinputs = None, 
        predictions = None,
    )
    task = NEW()
    task.stage = "TASK"
    
    with open(text_path, "r", encoding="utf-8") as f: 
        task.text = f.read().strip()
    return task

class PRE:
    def __call__(self, task):
        task.stage = "PRE"
        task.preinputs = f"Translate this from hindi to English:\nhindi: {task.text}\nEnglish:"
        del task.text 
        return task

class EXE:
    def __init__(self):
        self.model_id = "ModelSpace/GemmaX2-28-2B-v0.1"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            torch_dtype=torch.bfloat16,
            device_map="cpu"
        )

    def __call__(self, task):
        task.stage = "EXE"
        inputs = self.tokenizer(task.preinputs, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=50)
        task.predictions = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        del task.preinputs 
        return task

class POST:
    def __call__(self, task):
        task.stage = "POST"
        
        if "English:" in task.predictions:
            task.result = task.predictions.split("English:")[-1].strip()
        else:
            task.result = task.predictions
            
        del task.predictions
        return task