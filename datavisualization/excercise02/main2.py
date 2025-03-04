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
    pareto_set = pd.read_csv(f"MOEAD/DTLZ1_exe_{id}.ps", header=None)
    pareto_front = pd.read_csv(f"MOEAD/DTLZ1_exe_{id}.pf", header=None)
    all_pareto_sets.append(pareto_set)
    all_pareto_fronts.append(pareto_front)

pareto_set_np = np.vstack(all_pareto_sets)      # shape: (N, 6) - decision space
pareto_front_np = np.vstack(all_pareto_fronts)  # shape: (M, 2) - objective space

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

# Labels for decision variables
var_labels = [f"$x_{i}$" for i in range(pareto_set_np.shape[1])]

# =============================================================================
# Radar Chart (Spider Plot) for Decision Space (Full Pareto Set)
# =============================================================================
def normalize_values(values, min_val, max_val):
    """Normalize values between 0 and 1."""
    return (values - min_val) / (max_val - min_val)
# =============================================================================
# Pareto Front Visualization with Quantile Markers
# =============================================================================
sorted_front = pareto_front_np[np.argsort(pareto_front_np[:, 0])]  # Sort by Obj 1

q5_point = sorted_front[int(0.05 * len(sorted_front))]
q50_point = sorted_front[int(0.50 * len(sorted_front))]
q95_point = sorted_front[int(0.95 * len(sorted_front))]

plt.figure(figsize=(8, 6))
plt.scatter(sorted_front[:, 0], sorted_front[:, 1],s=10, color='gray', alpha=0.2, label='Pareto Points')
plt.plot(sorted_front[:, 0], sorted_front[:, 1], linestyle='-', color='black', alpha=0.1)

plt.scatter(q5_point[0], q5_point[1], color='red', s=100, label='5% Quantile')
plt.scatter(q50_point[0], q50_point[1], color='blue', s=100, label='50% Quantile (Median)')
plt.scatter(q95_point[0], q95_point[1], color='green', s=100, label='95% Quantile')

plt.xlabel("$f_0(x)$")
plt.ylabel("$f_1(x)$")
plt.title("Pareto Front with Quantile Markers")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# =============================================================================
# Parallel Coordinates Plot (Quantile Solutions)
# =============================================================================
def plot_parallel_coordinates(data, labels):
    """Creates a parallel coordinates plot for the 5%, 50%, and 95% quantile decision solutions."""
    num_vars = len(labels)

    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)

    fig, ax = plt.subplots(figsize=(10, 5))

    quantile_labels = ["5th Percentile", "50th Percentile", "95th Percentile"]
    line_styles = ["dashed", "solid", "dotted"]

    for i in range(3):
        ax.plot(range(num_vars), normalized_data[i], linestyle=line_styles[i], linewidth=2, label=quantile_labels[i])

    ax.set_xticks(range(num_vars))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Normalized Value", fontsize=12)
    ax.set_title("Decision Space: Parallel Coordinates of Quantile Solutions", fontsize=14)

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.show()

# Generate parallel coordinates plot for quantiles
plot_parallel_coordinates(quantile_solutions, var_labels)
