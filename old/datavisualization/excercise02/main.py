import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

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
# Radar Chart Function (Color by Rank or Cluster)
# =============================================================================
def normalize_values(values, min_val, max_val):
    """Normalize values between 0 and 1 for consistent radar chart scaling."""
    return (values - min_val) / (max_val - min_val)

def plot_radar_chart(data, labels, color_by="rank", num_samples=50):
    """Creates a radar chart for a given subset of Pareto set points."""
    num_vars = len(labels)
    
    # Compute the min and max for each variable
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)

    # Select a subset of random samples to plot
    sample_indices = np.random.choice(len(data), num_samples, replace=False)
    selected_samples = data[sample_indices]

    # Normalize data between 0 and 1
    normalized_data = np.array([
        normalize_values(selected_samples[:, i], min_vals[i], max_vals[i])
        for i in range(num_vars)
    ]).T  # Transpose so rows are samples

    # Create angles for radar chart axes
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Repeat first value to close the shape

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})

    # Choose colors based on the selected method
    if color_by == "rank":
        colors = sns.color_palette("coolwarm", as_cmap=True)(normalized_ranks[sample_indices])
    elif color_by == "cluster":
        colors = sns.color_palette("tab10", num_clusters)  # Assign a unique color to each cluster
        colors = [colors[cluster_labels[i]] for i in sample_indices]
    
    # Plot each sample with its corresponding color
    for i, sample in enumerate(normalized_data):
        values = sample.tolist()
        values += values[:1]  # Repeat first value to close the shape
        ax.plot(angles, values, linewidth=1, alpha=0.7, color=colors[i])
        ax.fill(angles, values, alpha=0.05, color=colors[i])

    # Configure axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # Set radial grid labels (0 to 1, normalized scale)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)

    title = "Radar Chart of Pareto Set (Color by " + ("Dominance Rank" if color_by == "rank" else "Cluster") + ")"
    plt.title(title, fontsize=14)
    plt.show()

# Labels for each decision variable
var_labels = [f"Var {i+1}" for i in range(pareto_set_np.shape[1])]

# Generate the radar chart (Choose "rank" for dominance rank, "cluster" for clustering)
#plot_radar_chart(pareto_set_np, var_labels, color_by="rank", num_samples=50)  # Change to "cluster" for clustering
plot_radar_chart(pareto_set_np, var_labels, color_by="cluster", num_samples=50)
