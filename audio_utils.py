import platform
import subprocess
import datetime
import numpy as np
from typing import Dict, List, Generator, Tuple

# Analogous to ffmpeg_read: Reads a static DAG file
def workflow_read(workflow_bytes: bytes, target_mips: float) -> Dict:
    """
    Helper function to parse a raw workflow payload. 
    Analogous to ffmpeg_read.
    """
    import json
    try:
        raw_data = json.loads(workflow_bytes.decode("utf-8"))
    except Exception as e:
        raise ValueError("Workflow payload is malformed or not valid JSON.") from e
    
    # "Resample" the tasks to the target MIPS capacity
    for task in raw_data.get("tasks", {}).values():
        task["v"] = task["v"] / target_mips # Scale cycles to time
    
    return raw_data

# Analogous to ffmpeg_microphone: Discovers compute nodes on the local network
def discover_compute_nodes(
    sampling_rate: int,
    discovery_timeout_s: float,
    ffmpeg_input_device: str | None = None
) -> Generator[bytes, None, None]:
    """
    Reads a stream of task requests from the local IoT bus.
    Uses 'alsa'-style logic to determine system platform.
    """
    system = platform.system()
    
    # Platform-specific discovery commands (Mocking ffmpeg-device logic)
    if system == "Linux":
        cmd = ["discovery-service", "--proto", "mqtt"]
    elif system == "Darwin": # MacOS
        cmd = ["discovery-service", "--proto", "bonjour"]
    elif system == "Windows":
        cmd = ["discovery-service", "--proto", "upnp"]
    
    # Internal stream simulation analogous to _ffmpeg_stream
    bufsize = 2**24
    try:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=bufsize) as proc:
            while True:
                task_raw = proc.stdout.read(1024) # Read task chunk
                if task_raw == b"":
                    break
                yield task_raw
    except FileNotFoundError:
        raise ValueError("Discovery service not found on system PATH.")

# Analogous to ffmpeg_microphone_live: Handles real-time overlapping task windows
def live_workflow_stream(
    sampling_rate: int,
    window_length_s: float,
    stride_length_s: float | None = None,
):
    """
    Streams tasks from the IoT device in overlapping windows.
    Makes use of striding to provide context to the Planner Agent.
    """
    # Initialize timing exactly like ffmpeg_microphone_live
    start_time = datetime.datetime.now()
    delta = datetime.timedelta(seconds=window_length_s)
    
    # Discovery generator
    stream = discover_compute_nodes(sampling_rate, window_length_s)
    
    # Mocking the chunk_bytes_iter logic to handle overlapping task windows
    stride = stride_length_s or (window_length_s / 6)
    
    for chunk in chunk_tasks_iter(stream, window_length_s, stride):
        # Time-sync check
        if datetime.datetime.now() > start_time + 10 * delta:
            # System is lagging, skip the current task window
            continue
            
        yield {
            "sampling_rate": sampling_rate,
            "raw_tasks": chunk,
            "partial": False
        }
        start_time += delta

def chunk_tasks_iter(iterator, chunk_len: float, stride: float):
    """
    Reads raw task bytes and creates chunks of length chunk_len with stride.
    Analogous to chunk_bytes_iter.
    """
    acc = []
    for item in iterator:
        acc.append(item)
        if len(acc) >= chunk_len:
            yield acc[:int(chunk_len)]
            # Sliding window logic using stride
            acc = acc[int(chunk_len - stride):]