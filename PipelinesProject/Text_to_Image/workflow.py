import os
import time
from workflows import tti as W 

# --- METRICS SETUP ---
step_count = 0
total_start = time.time()

def log_step(name):
    global step_count
    step_count += 1
    print(f"\n[Step {step_count}] {name}...")

def print_size(filename):
    size_kb = os.path.getsize(filename) / 1024
    print(f"   Buffer Size: {size_kb:.2f} KB")

# 1. TASK
log_step("TASK: Initialization")
task = W.TASK("input.txt")
W.Pack(task, buffer="tti_task.buffer")
print_size("tti_task.buffer")

# 2. PRE
log_step("PRE: Formatting Prompt")
task = W.Unpack("tti_task.buffer")
model_pre = W.PRE()
model_pre(task)
W.Pack(task, buffer="tti_task.buffer")
print_size("tti_task.buffer")

# 3. EXE
log_step("EXE: Image Generation")
print("   [EXE] Loading SD 3.5 Turbo... (First run may take time to download)")

exe_start = time.time()
task = W.Unpack("tti_task.buffer")
model_exe = W.EXE()
model_exe(task)
exe_end = time.time()

print(f"   [EXE] Done! Execution Time: {exe_end - exe_start:.4f} seconds")
W.Pack(task, buffer="tti_task.buffer")
print_size("tti_task.buffer")

# 4. POST
log_step("POST: Image Export")
task = W.Unpack("tti_task.buffer")
model_post = W.POST()
model_post(task)
print("\n   Final Status:", task.result)

# Final Summary
print("\n" + "="*30)
print(f"Total Time: {time.time() - total_start:.4f} s")
print("="*30)