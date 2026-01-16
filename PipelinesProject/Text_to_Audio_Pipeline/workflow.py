import os
import time
from workflows import tta as W

step_count = 0
total_start = time.time()

def log_step(name):
    global step_count
    step_count += 1
    print(f"\n[Step {step_count}] {name}...")

def print_size(filename):
    size_kb = os.path.getsize(filename) / 1024
    print(f"   Buffer Size: {size_kb:.2f} KB")

# 1. Create the task 
log_step("TASK: Initialization")
task = W.TASK("input.txt")
W.Pack(task, buffer="tta_task.buffer")
print_size("tta_task.buffer")

# 2. Pre-process 
log_step("PRE: Formatting Prompt")
task = W.Unpack("tta_task.buffer")
model_pre = W.PRE()
model_pre(task)
W.Pack(task, buffer="tta_task.buffer")
print_size("tta_task.buffer")

# 3. Execute (Runs MusicGen)
log_step("EXE: Audio Generation")
print("   [EXE] Loading MusicGen and Generating... (This may take a moment)")

exe_start = time.time()
task = W.Unpack("tta_task.buffer")
model_exe = W.EXE()
model_exe(task)
exe_end = time.time()

exe_duration = exe_end - exe_start
print(f"   [EXE] Done! Execution Time: {exe_duration:.4f} seconds")
W.Pack(task, buffer="tta_task.buffer")
print_size("tta_task.buffer")

# 4. Post-process 
log_step("POST: Audio Export")
task = W.Unpack("tta_task.buffer")
model_post = W.POST()
model_post(task)
print("\n   Final Status:", task.result)

# Final Summary
total_duration = time.time() - total_start
print("\n")
print("      TTA RESULTS SUMMARY")
print(f"Total Steps : {step_count}")
print(f"Model Time  : {exe_duration:.4f} s")
print(f"Total Time  : {total_duration:.4f} s")