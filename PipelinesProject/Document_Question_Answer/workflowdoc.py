import os
import time
from workflows import docqa as W

def print_size(filename):
    if os.path.exists(filename):
        size_bytes = os.path.getsize(filename)
        if size_bytes < 1024 * 1024:
            print(f"   [Tx] Data Transfer: {size_bytes/1024:.2f} KB")
        else:
            print(f"   [Tx] Data Transfer: {size_bytes/(1024*1024):.2f} MB")

class MetricTracker:
    def __init__(self):
        self.step = 0
    def tick(self, stage, desc):
        self.step += 1
        print(f"\n [Step {self.step}] {stage}: {desc}")

def run_pipeline():
    tracker = MetricTracker()
    start_time = time.time()
    
    # 1. TASK GENERATION
    tracker.tick("TASK", "Load Data")
    task = W.TASK("context.txt", "question.txt")
    
    W.Pack(task, buffer="task.buffer")
    print_size("task.buffer")
    task = W.Unpack("task.buffer")
    
    # 2. PRE-PROCESSING
    tracker.tick("PRE", "Format Prompt")
    model_pre = W.PRE()
    model_pre(task)
    
    W.Pack(task, buffer="task.buffer")
    print_size("task.buffer")
    task = W.Unpack("task.buffer")
    
    # 3. EXECUTION (Inference)
    tracker.tick("EXE", "Inference")
    model_exe = W.EXE()
    model_exe(task)
    
    W.Pack(task, buffer="task.buffer")
    print_size("task.buffer")
    task = W.Unpack("task.buffer")
    
    # 4. POST-PROCESSING
    tracker.tick("POST", "Formatted Result")
    model_post = W.POST()
    model_post(task)
    print(f"    Result: {task.result}")
    
    W.Pack(task, buffer="task.buffer")
    print_size("task.buffer")
    
    total_time = time.time() - start_time
    return total_time, tracker.step

if __name__ == "__main__":
    if not os.path.exists("context.txt"):
        with open("context.txt", "w", encoding="utf-8") as f: 
            f.write("Agentic AI is a class of artificial intelligence that can take independent action to achieve goals. Unlike passive AI tools, agents can perceive, reason, and act.")
    
    if not os.path.exists("question.txt"):
        with open("question.txt", "w", encoding="utf-8") as f: 
            f.write("What can Agentic AI do?")

    t_pipe, s_pipe = run_pipeline()

    # Results
    print(f"{'Metric':<20} | {'Pipeline':<10} ")
    print(f"{'Time (s)':<20} | {t_pipe:<10.2f}")
    print(f"{'Steps (Count)':<20} | {s_pipe:<10}")