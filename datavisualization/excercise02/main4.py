import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# =============================================================================
# Load Pareto Set & Pareto Front Data
# =============================================================================
all_pareto_sets = []
all_pareto_fronts = []

for id in range(1, 31):
    pareto_front = pd.read_csv(f"MOEAD/DTLZ1_exe_{id}.pf", header=None)
    pareto_set = pd.read_csv(f"MOEAD/DTLZ1_exe_{id}.ps", header=None)

    all_pareto_sets.append(pareto_set)
    all_pareto_fronts.append(pareto_front)

pareto_set_np = np.vstack(all_pareto_sets)    # shape: (N, 6)
pareto_front_np = np.vstack(all_pareto_fronts)  # shape: (N, 2)

# =============================================================================
# Compute Quantiles (5%, 50%, 95%) based on Pareto Front
# =============================================================================
dominance_sums = np.sum(pareto_front_np, axis=1)  # A simple dominance approximation

sorted_indices = np.argsort(dominance_sums)
q5_idx = sorted_indices[int(0.05 * len(sorted_indices))]
q50_idx = sorted_indices[int(0.50 * len(sorted_indices))]
q95_idx = sorted_indices[int(0.95 * len(sorted_indices))]
quantile_indices = [q5_idx, q50_idx, q95_idx]

# Get the corresponding decision variable values from the Pareto Set
quantile_solutions = pareto_set_np[quantile_indices]

# =============================================================================
# Radar Chart (Quantile Solutions)
# =============================================================================
def plot_radar_chart(data, labels):
    """Creates a radar chart for the 5%, 50%, and 95% quantile decision solutions."""
    num_vars = len(labels)
    
    # Normalize values for radar chart
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    normalized_data = (data - min_vals) / (max_vals - min_vals)

    # Create angles for radar chart axes
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Repeat first value to close the shape

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})

    # Plot each quantile solution
    quantile_labels = ["5th Percentile", "50th Percentile", "95th Percentile"]
    line_styles = ["dashed", "solid", "dotted"]

    for i in range(3):
        values = normalized_data[i].tolist()
        values += values[:1]  # Repeat first value to close the shape
        ax.plot(angles, values, linestyle=line_styles[i], linewidth=2, label=quantile_labels[i])

    # Configure axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)

    # Set radial grid labels (0 to 1, normalized scale)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)

    plt.title("Decision Space: Radar Chart of Quantile Solutions", fontsize=14)
    plt.legend(loc="upper right", fontsize=10)
    plt.show()

# Labels for each decision variable
var_labels = [f"Var {i+1}" for i in range(pareto_set_np.shape[1])]

# Generate radar chart for quantiles
plot_radar_chart(quantile_solutions, var_labels)

# =============================================================================
# Parallel Coordinates Plot (Quantile Solutions)
# =============================================================================
def plot_parallel_coordinates(data, labels):
    """Creates a parallel coordinates plot for the 5%, 50%, and 95% quantile decision solutions."""
    num_vars = len(labels)

    # Normalize all data to [0, 1] for consistent visualization
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot quantile solutions
    quantile_labels = ["5th Percentile", "50th Percentile", "95th Percentile"]
    line_styles = ["dashed", "solid", "dotted"]

    for i in range(3):
        ax.plot(range(num_vars), normalized_data[i], linestyle=line_styles[i], linewidth=2, label=quantile_labels[i])

    # Axis labels and ticks
    ax.set_xticks(range(num_vars))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Normalized Value", fontsize=12)
    ax.set_title("Decision Space: Parallel Coordinates of Quantile Solutions", fontsize=14)

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.show()

# Generate parallel coordinates plot for quantiles
plot_parallel_coordinates(quantile_solutions, var_labels)
