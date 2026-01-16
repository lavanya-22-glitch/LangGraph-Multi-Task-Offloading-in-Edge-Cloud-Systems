from workflows import Fake, Pack, Unpack
import torch
from diffusers import StableDiffusion3Pipeline


# ... rest of your TTI logic ...

def TASK(text_path):
    def NEW(): return Fake(
        pipe = "TTI", 
        stage = None,
        image = None, 
        text = None,
        preinputs = None, 
    )
    task = NEW()
    task.stage = "TASK"
    with open(text_path, "r") as f: task.text = f.read()
    return task

class PRE:
    def __call__(self, task):
        task.stage = "PRE"
        # Standard text prompt for Stable Diffusion
        task.preinputs = task.text
        del task.text 
        return task

class EXE:
    def __init__(self):
        self.model_id = "tensorart/stable-diffusion-3.5-medium-turbo"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # SD 3.5 is heavy; use bfloat16 for better CPU performance if available
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            self.model_id, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.bfloat16
        )
        self.pipe.to(self.device)
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()

    def __call__(self, task):
        task.stage = "EXE"
        # Turbo models only need a few steps
        result = self.pipe(
            prompt=task.preinputs,
            num_inference_steps=4, # Lowering to 4 for faster testing
            guidance_scale=0.0    # Turbo models often use 0.0 or low guidance
        ).images[0]
        
        task.image = result
        del task.preinputs 
        return task

class POST:
    def __call__(self, task):
        task.stage = "POST"
        
        output_filename = "output_image.png"
        task.image.save(output_filename)
        
        task.result = f"Image saved to {output_filename}"
        del task.image
        return task