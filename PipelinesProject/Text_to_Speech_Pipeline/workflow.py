import os
import time
from workflows import tts as W

# --- METRICS SETUP ---
step_count = 0
total_start = time.time()

def log_step(name):
    global step_count
    step_count += 1
    print(f"\n[Step {step_count}] {name}...")

# Helper function to print size nicely
def print_size(filename):
    size_kb = os.path.getsize(filename) / 1024
    print(f"   Size of {filename}: {size_kb:.2f} KB")

# 1. Create the task 
log_step("TASK: Initialization")
task = W.TASK("input.txt")
print("   Task Created:", task.__dict__)

# Measure size
W.Pack(task, buffer="task.buffer")
print_size("task.buffer")

# 2. Pre-process 
log_step("PRE: Formatting")
task = W.Unpack("task.buffer")
model_pre = W.PRE()
model_pre(task)
print("   After PRE:", task.__dict__)

# Measure size
W.Pack(task, buffer="task.buffer")
print_size("task.buffer")

# 3. Execute (Runs the Llama model)
log_step("EXE: Inference")
print("   [EXE] Loading Model and Running... (Timer Started)")

exe_start = time.time() # Start Timer for Model Only
task = W.Unpack("task.buffer")
model_exe = W.EXE()
model_exe(task)
exe_end = time.time()   # Stop Timer

exe_duration = exe_end - exe_start
print(f"   [EXE] Done! Execution Time: {exe_duration:.4f} seconds")
print("   After EXE:", task.__dict__)

# Measure size
W.Pack(task, buffer="task.buffer")
print_size("task.buffer")

# 4. Post-process 
log_step("POST: Cleanup")
task = W.Unpack("task.buffer")
model_post = W.POST()
model_post(task)
print("\n   Final Result:", task.result)

# Final measure
W.Pack(task, buffer="task.buffer")
print_size("task.buffer")

# --- FINAL SUMMARY ---
total_duration = time.time() - total_start
print("\n" + "="*30)
print("      RESULTS SUMMARY")
print("="*30)
print(f"Total Steps : {step_count}")
print(f"Model Time  : {exe_duration:.4f} s")
print(f"Total Time  : {total_duration:.4f} s")
print("="*30)