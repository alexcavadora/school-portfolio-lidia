import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from matplotlib.collections import LineCollection

# =============================================================================
# Load Pareto Set Data
# =============================================================================
all_pareto_sets = []
for id in range(1, 31):
    pareto_set = pd.read_csv(f"MOEAD/DTLZ1_exe_{id}.ps", header=None)
    all_pareto_sets.append(pareto_set)

pareto_set_np = np.vstack(all_pareto_sets)  # shape: (N, 6)

# =============================================================================
# Compute Dominance Rank
# =============================================================================
def compute_dominance_rank(data):
    """Computes dominance rank based on Pareto dominance"""
    ranks = np.zeros(len(data))
    for i in range(len(data)):
        for j in range(len(data)):
            if i != j and np.all(data[j] >= data[i]) and np.any(data[j] > data[i]):
                ranks[i] += 1  # More dominated points get a higher rank
    return ranks

dominance_ranks = compute_dominance_rank(pareto_set_np)

# Normalize dominance ranks for color mapping
normalized_ranks = (dominance_ranks - dominance_ranks.min()) / (dominance_ranks.max() - dominance_ranks.min())

# =============================================================================
# Clustering: K-Means
# =============================================================================
num_clusters = 4  # Adjust the number of clusters as needed
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(pareto_set_np)

# =============================================================================
# Parallel Coordinates Plot Function
# =============================================================================
def plot_parallel_coordinates(data, labels, color_by="rank", num_samples=200):
    """Creates a parallel coordinates plot for Pareto set solutions"""
    
    num_vars = len(labels)
    
    # Select a subset of samples to avoid overcrowding
    sample_indices = np.random.choice(len(data), num_samples, replace=False)
    selected_samples = data[sample_indices]

    # Normalize all data to [0, 1] for consistent visualization
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(selected_samples)

    # Choose colors based on method
    if color_by == "rank":
        colors = sns.color_palette("coolwarm", as_cmap=True)(normalized_ranks[sample_indices])
    elif color_by == "cluster":
        colors = sns.color_palette("tab10", num_clusters)
        colors = [colors[cluster_labels[i]] for i in sample_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate parallel coordinates plot
    for i, (line, color) in enumerate(zip(normalized_data, colors)):
        ax.plot(range(num_vars), line, color=color, alpha=0.5, linewidth=1.2)

    # Axis labels and ticks
    ax.set_xticks(range(num_vars))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Normalized Value", fontsize=12)
    ax.set_title("Parallel Coordinates Plot (Color by " + ("Dominance Rank" if color_by == "rank" else "Cluster") + ")", fontsize=14)

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

# Labels for decision variables
var_labels = [f"Var {i+1}" for i in range(pareto_set_np.shape[1])]

# Generate parallel coordinates plot (Choose "rank" for dominance rank, "cluster" for clustering)
plot_parallel_coordinates(pareto_set_np, var_labels, color_by="cluster", num_samples=200)  # Change to "cluster" for clustering
