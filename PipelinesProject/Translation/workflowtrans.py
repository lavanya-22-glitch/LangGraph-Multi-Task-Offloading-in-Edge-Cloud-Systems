import os
import time
from workflows import translation as W

def print_size(filename):
    if os.path.exists(filename):
        size_bytes = os.path.getsize(filename)
        print(f"   [Tx] Data Transfer: {size_bytes/1024:.2f} KB")

class MetricTracker:
    def __init__(self):
        self.step = 0
    def tick(self, stage, desc):
        self.step += 1
        print(f"\n [Step {self.step}] {stage}: {desc}")

def run_pipeline():
    tracker = MetricTracker()
    start_time = time.time()
    
    # 1. TASK
    tracker.tick("TASK", "Load Hindi Text")
    task = W.TASK("input.txt")
    
    W.Pack(task, buffer="task.buffer")
    print_size("task.buffer")
    task = W.Unpack("task.buffer")
    
    # 2. PRE
    tracker.tick("PRE", "Format Translation Prompt")
    model_pre = W.PRE()
    model_pre(task)
    
    W.Pack(task, buffer="task.buffer")
    print_size("task.buffer")
    task = W.Unpack("task.buffer")
    
    # 3. EXE
    tracker.tick("EXE", "Inference (GemmaX2 Model)")
    model_exe = W.EXE()
    model_exe(task)
    
    W.Pack(task, buffer="task.buffer")
    print_size("task.buffer")
    task = W.Unpack("task.buffer")
    
    # 4. POST
    tracker.tick("POST", "Extract English Translation")
    model_post = W.POST()
    model_post(task)
    print(f"    Result: {task.result}")
    
    total_time = time.time() - start_time
    return total_time, tracker.step

if __name__ == "__main__":
    if not os.path.exists("input.txt"):
        with open("input.txt", "w", encoding="utf-8") as f: 
            f.write("haan mei yaha hu")

    t_pipe, s_pipe = run_pipeline()

    print(f"{'Metric':<20} | {'Translation':<10} ")
    print(f"{'Time (s)':<20} | {t_pipe:<10.2f}")
    print(f"{'Steps (Count)':<20} | {s_pipe:<10}")