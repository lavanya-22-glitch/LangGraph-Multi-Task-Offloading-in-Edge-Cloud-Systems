# memory_manager.py - Memory system for storing and retrieving historical runs
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from datetime import datetime


class WorkflowMemory:
    """
    Stores and retrieves historical workflow execution data for few-shot prompting.
    Each execution is stored as a separate JSON file in a memory directory.
    """
    
    def __init__(self, memory_dir: str = "memory_store"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        
    def extract_workflow_features(self, workflow_dict: dict) -> dict:
        """
        Extract numerical and structural features from a workflow.
        
        Returns:
            Dictionary with workflow characteristics
        """
        tasks = workflow_dict.get('tasks', {})
        edges = workflow_dict.get('edges', {})
        N = workflow_dict.get('N', 0)
        
        if N == 0 or not tasks:
            return {}
        
        # Compute size statistics
        v_values = [t.get('v', 0) for t in tasks.values()]
        v_mean = np.mean(v_values) if v_values else 0
        v_std = np.std(v_values) if v_values else 0
        v_max = max(v_values) if v_values else 0
        v_min = min(v_values) if v_values else 0
        
        # Edge statistics
        edge_count = len(edges)
        max_possible_edges = N * (N - 1)
        edge_density = edge_count / max_possible_edges if max_possible_edges > 0 else 0
        
        edge_bytes = list(edges.values())
        d_mean = np.mean(edge_bytes) if edge_bytes else 0
        d_std = np.std(edge_bytes) if edge_bytes else 0
        d_max = max(edge_bytes) if edge_bytes else 0
        d_min = min(edge_bytes) if edge_bytes else 0
        
        # Compute in-degree and out-degree distributions
        in_degree = {i: 0 for i in range(1, N + 1)}
        out_degree = {i: 0 for i in range(1, N + 1)}
        
        for (src, dst) in edges.keys():
            if 1 <= src <= N:
                out_degree[src] += 1
            if 1 <= dst <= N:
                in_degree[dst] += 1
        
        # Count nodes by degree patterns
        nodes_in1_out1 = sum(1 for i in range(1, N + 1) if in_degree[i] == 1 and out_degree[i] == 1)
        nodes_source = sum(1 for i in range(1, N + 1) if in_degree[i] == 0)
        nodes_sink = sum(1 for i in range(1, N + 1) if out_degree[i] == 0)
        
        # Approximate critical path length (simple heuristic: longest path in terms of edges)
        cp_length = self._estimate_critical_path_length(N, edges)
        
        return {
            "N": N,
            "v_mean": float(v_mean),
            "v_std": float(v_std),
            "v_max": float(v_max),
            "v_min": float(v_min),
            "edge_count": edge_count,
            "edge_density": float(edge_density),
            "d_mean": float(d_mean),
            "d_std": float(d_std),
            "d_max": float(d_max),
            "d_min": float(d_min),
            "nodes_linear": nodes_in1_out1,
            "nodes_source": nodes_source,
            "nodes_sink": nodes_sink,
            "critical_path_length": cp_length
        }
    
    def _estimate_critical_path_length(self, N: int, edges: dict) -> int:
        """
        Estimate critical path length using longest path in DAG (by edge count).
        Simple BFS/DFS based approach.
        """
        # Build adjacency list
        adj = {i: [] for i in range(0, N + 2)}
        for (src, dst) in edges.keys():
            adj[src].append(dst)
        
        # Compute longest path from node 0 (entry) using topological sort
        in_degree = {i: 0 for i in range(0, N + 2)}
        for src in adj:
            for dst in adj[src]:
                in_degree[dst] += 1
        
        # Topological sort with distance tracking
        from collections import deque
        queue = deque([i for i in range(0, N + 2) if in_degree[i] == 0])
        dist = {i: 0 for i in range(0, N + 2)}
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                dist[v] = max(dist[v], dist[u] + 1)
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        
        return dist.get(N + 1, 0)
    
    def extract_environment_features(self, env_dict: dict) -> dict:
        """
        Extract numerical features from environment configuration.
        
        Returns:
            Dictionary with environment characteristics
        """
        locations = env_dict.get('locations', {})
        DR_map = env_dict.get('DR', {})
        DE_map = env_dict.get('DE', {})
        VR_map = env_dict.get('VR', {})
        VE_map = env_dict.get('VE', {})
        
        # Count location types
        iot_count = sum(1 for t in locations.values() if t == 'iot')
        edge_count = sum(1 for t in locations.values() if t == 'edge')
        cloud_count = sum(1 for t in locations.values() if t == 'cloud')
        
        # DR statistics (exclude self-loops)
        dr_values = [v for (src, dst), v in DR_map.items() if src != dst and v < float('inf')]
        dr_mean = np.mean(dr_values) if dr_values else 0
        dr_std = np.std(dr_values) if dr_values else 0
        dr_max = max(dr_values) if dr_values else 0
        dr_min = min(dr_values) if dr_values else 0
        
        # DE statistics
        de_values = list(DE_map.values())
        de_mean = np.mean(de_values) if de_values else 0
        de_std = np.std(de_values) if de_values else 0
        
        # VR statistics
        vr_values = list(VR_map.values())
        vr_mean = np.mean(vr_values) if vr_values else 0
        vr_std = np.std(vr_values) if vr_values else 0
        
        # VE statistics
        ve_values = list(VE_map.values())
        ve_mean = np.mean(ve_values) if ve_values else 0
        ve_std = np.std(ve_values) if ve_values else 0
        
        # Regime detection (simple heuristic)
        regime = self._detect_regime(dr_mean, vr_mean, ve_mean)
        
        return {
            "location_count": len(locations),
            "iot_count": iot_count,
            "edge_count": edge_count,
            "cloud_count": cloud_count,
            "dr_mean": float(dr_mean),
            "dr_std": float(dr_std),
            "dr_max": float(dr_max),
            "dr_min": float(dr_min),
            "de_mean": float(de_mean),
            "de_std": float(de_std),
            "vr_mean": float(vr_mean),
            "vr_std": float(vr_std),
            "ve_mean": float(ve_mean),
            "ve_std": float(ve_std),
            "regime": regime
        }
    
    def _detect_regime(self, dr_mean: float, vr_mean: float, ve_mean: float) -> str:
        """
        Detect whether system is network-bound, compute-bound, or energy-bound.
        Simple heuristic based on order of magnitude comparison.
        """
        # Normalize to same scale (roughly)
        # DR: ms/byte, VR: ms/cycle, VE: mJ/cycle
        # Higher DR -> network-bound
        # Higher VR -> compute-bound
        # Higher VE -> energy-bound
        
        if dr_mean > 1e-4:  # High network latency
            return "network-bound"
        elif vr_mean > 1e-7:  # Slow computation
            return "compute-bound"
        elif ve_mean > 5e-7:  # High energy cost
            return "energy-bound"
        else:
            return "balanced"
    
    def generate_textual_summary(self, workflow_features: dict, env_features: dict, 
                                 params: dict) -> str:
        """
        Generate a human-readable summary of the scenario for LLM prompting.
        """
        wf = workflow_features
        env = env_features
        
        summary_lines = []
        
        # Workflow summary
        summary_lines.append(f"Workflow: {wf['N']} tasks")
        summary_lines.append(f"  - Avg compute: {wf['v_mean']:.2e} cycles (std: {wf['v_std']:.2e})")
        summary_lines.append(f"  - Edge density: {wf['edge_density']:.2%} ({wf['edge_count']} edges)")
        summary_lines.append(f"  - Avg data dependency: {wf['d_mean']:.2e} bytes")
        summary_lines.append(f"  - Critical path length: ~{wf['critical_path_length']} hops")
        
        if wf['nodes_linear'] > wf['N'] * 0.5:
            summary_lines.append(f"  - Structure: Mostly linear chain")
        elif wf['nodes_source'] > 2 and wf['nodes_sink'] > 2:
            summary_lines.append(f"  - Structure: Multiple sources/sinks (parallel)")
        else:
            summary_lines.append(f"  - Structure: Mixed topology")
        
        # Environment summary
        summary_lines.append(f"Environment: {env['location_count']} locations")
        summary_lines.append(f"  - {env['edge_count']} edge, {env['cloud_count']} cloud servers")
        summary_lines.append(f"  - Avg network latency: {env['dr_mean']:.2e} ms/byte")
        summary_lines.append(f"  - Avg compute speed: {env['vr_mean']:.2e} ms/cycle")
        summary_lines.append(f"  - Avg task energy: {env['ve_mean']:.2e} mJ/cycle")
        summary_lines.append(f"  - Regime: {env['regime']}")
        
        # Mode
        delta_t = params.get('delta_t', 1)
        delta_e = params.get('delta_e', 1)
        if delta_t == 1 and delta_e == 1:
            mode = "BALANCED"
        elif delta_t == 1 and delta_e == 0:
            mode = "LOW LATENCY"
        elif delta_t == 0 and delta_e == 1:
            mode = "LOW POWER"
        else:
            mode = "CUSTOM"
        summary_lines.append(f"Mode: {mode}")
        
        return "\n".join(summary_lines)
    
    def save_execution(self, workflow_dict: dict, env_dict: dict, params: dict,
                      optimal_policy: List[int], evaluation_result: dict,
                      plan: str, experiment_id: str) -> str:
        """
        Save a complete execution record to a JSON file.
        
        Returns:
            Path to the saved memory file
        """
        # Extract features
        wf_features = self.extract_workflow_features(workflow_dict)
        env_features = self.extract_environment_features(env_dict)
        
        # Generate summary
        summary = self.generate_textual_summary(wf_features, env_features, params)
        
        # Create memory record
        memory_record = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "workflow_features": wf_features,
            "environment_features": env_features,
            "params": params,
            "optimal_policy": optimal_policy,
            "best_cost": evaluation_result.get("best_cost"),
            "evaluated_policies": evaluation_result.get("evaluated"),
            "skipped_policies": evaluation_result.get("skipped"),
            "summary": summary,
            "plan_excerpt": plan[:500] if plan else ""
        }
        
        # Save to file
        filename = f"memory_{experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.memory_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(memory_record, f, indent=2)
        
        print(f"💾 Saved execution memory to: {filepath}")
        return str(filepath)
    
    def compute_feature_vector(self, wf_features: dict, env_features: dict) -> np.ndarray:
        """
        Combine workflow and environment features into a single normalized vector.
        """
        # Select key features (ensure same order every time)
        features = [
            wf_features.get('N', 0),
            math.log10(wf_features.get('v_mean', 1) + 1),  # Log scale for large values
            wf_features.get('edge_density', 0),
            math.log10(wf_features.get('d_mean', 1) + 1),
            wf_features.get('critical_path_length', 0) / max(wf_features.get('N', 1), 1),  # Normalize
            env_features.get('location_count', 0),
            math.log10(env_features.get('dr_mean', 1e-10) + 1e-10),
            math.log10(env_features.get('vr_mean', 1e-10) + 1e-10),
            math.log10(env_features.get('ve_mean', 1e-10) + 1e-10),
        ]
        
        return np.array(features, dtype=float)
    
    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute similarity between two feature vectors.
        Uses normalized Euclidean distance (lower = more similar).
        """
        # Avoid division by zero
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return float('inf')
        
        # Normalized Euclidean distance
        distance = np.linalg.norm(vec1 - vec2)
        normalized_distance = distance / (norm1 + norm2)
        
        return normalized_distance
    
    def retrieve_similar_executions(self, workflow_dict: dict, env_dict: dict,
                                    params: dict, top_k: int = 3) -> List[dict]:
        """
        Retrieve top-k most similar past executions for few-shot prompting.
        
        Args:
            workflow_dict: Current workflow
            env_dict: Current environment
            params: Current parameters
            top_k: Number of similar examples to retrieve
            
        Returns:
            List of memory records sorted by similarity
        """
        # Extract features for current scenario
        current_wf_features = self.extract_workflow_features(workflow_dict)
        current_env_features = self.extract_environment_features(env_dict)
        current_vector = self.compute_feature_vector(current_wf_features, current_env_features)
        
        # Load all memory files
        memory_files = list(self.memory_dir.glob("memory_*.json"))
        
        if not memory_files:
            print("📭 No historical executions found in memory.")
            return []
        
        print(f"🔍 Searching through {len(memory_files)} historical executions...")
        
        # Compute similarity for each memory
        similarities = []
        for filepath in memory_files:
            try:
                with open(filepath, 'r') as f:
                    memory_record = json.load(f)
                
                # Skip if mode is different (optional filter)
                if memory_record.get('params', {}).get('delta_t') != params.get('delta_t') or \
                   memory_record.get('params', {}).get('delta_e') != params.get('delta_e'):
                    continue
                
                # Skip if N is very different (optional: within 30% range)
                stored_N = memory_record.get('workflow_features', {}).get('N', 0)
                current_N = current_wf_features.get('N', 0)
                if current_N > 0 and abs(stored_N - current_N) / current_N > 0.3:
                    continue
                
                # Compute feature vector for stored execution
                stored_vector = self.compute_feature_vector(
                    memory_record['workflow_features'],
                    memory_record['environment_features']
                )
                
                # Compute similarity
                similarity = self.compute_similarity(current_vector, stored_vector)
                
                similarities.append({
                    'filepath': str(filepath),
                    'similarity': similarity,
                    'memory_record': memory_record
                })
            except Exception as e:
                print(f"⚠️  Error loading {filepath}: {e}")
                continue
        
        # Sort by similarity (lower is better)
        similarities.sort(key=lambda x: x['similarity'])
        
        # Return top-k
        top_similar = similarities[:top_k]
        
        if top_similar:
            print(f"✅ Found {len(top_similar)} similar executions:")
            for i, item in enumerate(top_similar, 1):
                exp_id = item['memory_record'].get('experiment_id', 'unknown')
                sim_score = item['similarity']
                print(f"  {i}. Experiment {exp_id} (similarity: {sim_score:.4f})")
        
        return [item['memory_record'] for item in top_similar]
    
    def format_few_shot_examples(self, similar_executions: List[dict]) -> str:
        """
        Format retrieved similar executions into few-shot prompt text.
        """
        if not similar_executions:
            return ""
        
        few_shot_text = []
        few_shot_text.append("## Historical Similar Cases for Reference:\n")
        
        for i, memory in enumerate(similar_executions, 1):
            few_shot_text.append(f"### Example {i}:")
            few_shot_text.append(memory['summary'])
            few_shot_text.append(f"\nOptimal Policy Found: {memory['optimal_policy']}")
            few_shot_text.append(f"Total Cost: {memory.get('best_cost', 'N/A'):.6f}")
            
            # Add strategic insight if available
            if memory.get('plan_excerpt'):
                few_shot_text.append(f"\nKey Strategy: {memory['plan_excerpt'][:200]}...")
            
            few_shot_text.append("\n" + "-"*60 + "\n")
        
        return "\n".join(few_shot_text)