
from . import Fake, Pack, Unpack

def TASK(image_path, text_path):

    def NEW(): return Fake(
        pipe = "VQA",
        stage = None,

        image = None,
        text = None,
        
        preinputs = None,
        predictions = None,
    )

    task = NEW()
    task.stage = "TASK"
    from PIL import Image
    task.image = Image.open(image_path).convert("RGB")
    with open(text_path, "r") as f: task.text = f.read()
    return task

class PRE:
    from transformers import Pix2StructProcessor
    def __init__(self): self.processor = __class__.Pix2StructProcessor.from_pretrained("google/pix2struct-ai2d-base")
    def __call__(self, task): 
        task.stage = "PRE"
        task.preinputs = self.processor(images=task.image, text=task.text, return_tensors="pt")
        del task.image, task.text
        return task 

class EXE:
    from transformers import Pix2StructForConditionalGeneration
    def __init__(self): self.model = __class__.Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-ai2d-base")
    def __call__(self, task): 
        task.stage = "PRE"
        task.predictions = self.model.generate(**task.preinputs)
        del task.preinputs
        return task

class POST:
    from transformers import Pix2StructProcessor
    def __init__(self): self.processor= __class__.Pix2StructProcessor.from_pretrained("google/pix2struct-ai2d-base")
    def __call__(self, task): 
        task.stage = "POST"
        task.result = [self.processor.decode(i, skip_special_tokens=True) for i in task.predictions]
        del task.predictions
        return  task


        