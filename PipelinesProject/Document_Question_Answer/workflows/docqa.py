from . import Fake, Pack, Unpack
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

def TASK(context_path, question_path):
    def NEW(): return Fake(
        pipe = "DOCQA", 
        stage = None,
        context = None, 
        question = None,
        inputs = None, 
        predictions = None,
    )
    task = NEW()
    task.stage = "TASK"
    
    with open(context_path, "r", encoding="utf-8") as f: task.context = f.read()
    with open(question_path, "r", encoding="utf-8") as f: task.question = f.read()
    return task

class PRE:
    def __call__(self, task):
        task.stage = "PRE"
        return task

class EXE:
    def __init__(self):
        self.model_id = "deepset/roberta-base-squad2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_id)

    def __call__(self, task):
        task.stage = "EXE"
        inputs = self.tokenizer(task.question, task.context, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        
        if answer_end < answer_start:
            task.predictions = "I could not find the answer in the text."
        else:
            answer_tokens = inputs.input_ids[0, answer_start:answer_end]
            # skip_special_tokens=True removes garbage like [SEP] or <s>
            task.predictions = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        return task

class POST:
    def __call__(self, task):
        task.stage = "POST"
        task.result = task.predictions.strip()
        
        # Cleanup
        del task.predictions, task.context, task.question
        return task