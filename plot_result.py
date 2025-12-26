import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION ---
FILE_BASELINE = "results_baseline.csv"  # Rename your first run file to this
FILE_MEMORY = "results_memory.csv"      # Rename your second run file to this
OUTPUT_IMAGE = "impact_of_memory.png"

def plot_comparison():
    if not os.path.exists(FILE_BASELINE) or not os.path.exists(FILE_MEMORY):
        print(f"❌ Missing files! Please ensure '{FILE_BASELINE}' and '{FILE_MEMORY}' exist.")
        return

    # 1. Load Data
    df_base = pd.read_csv(FILE_BASELINE)
    df_mem = pd.read_csv(FILE_MEMORY)

    # 2. Merge on Scenario Name to align them
    # We use 'inner' merge to ensure we only compare matching scenarios
    df_merged = pd.merge(df_base, df_mem, on='Scenario', suffixes=('_Baseline', '_Memory'))

    # 3. Setup Plot
    plt.figure(figsize=(14, 7))
    x = np.arange(len(df_merged['Scenario']))
    width = 0.35  # Width of the bars

    # 4. Create Bars
    # Baseline (Grey)
    plt.bar(x - width/2, df_merged['Agent_Cost_Baseline'], width, 
            label='Baseline (Cold Run)', color='#95a5a6', edgecolor='black')
    
    # Memory (Green - Shows Improvement)
    plt.bar(x + width/2, df_merged['Agent_Cost_Memory'], width, 
            label='With Memory (ICL)', color='#2ecc71', edgecolor='black')

    # 5. Styling
    plt.xlabel('Network Scenarios', fontweight='bold', fontsize=12)
    plt.ylabel('Total Cost (Time + Energy)', fontweight='bold', fontsize=12)
    plt.title('Ablation Study: Impact of In-Context Learning', fontweight='bold', fontsize=14)
    plt.xticks(x, df_merged['Scenario'], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add Cost Reduction Percentage Labels
    for i in range(len(df_merged)):
        base = df_merged['Agent_Cost_Baseline'][i]
        mem = df_merged['Agent_Cost_Memory'][i]
        if base > 0:
            diff = ((base - mem) / base) * 100
            if diff > 0.1: # Only show if there's a visible improvement
                plt.text(x[i] + width/2, mem + (mem*0.02), f"-{diff:.1f}%", 
                         ha='center', va='bottom', fontsize=8, color='green', fontweight='bold')

    plt.tight_layout()

    # 6. Save
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"✅ Chart saved to: {OUTPUT_IMAGE}")
    plt.show()

if __name__ == "__main__":
    plot_comparison()