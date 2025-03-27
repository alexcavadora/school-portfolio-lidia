#%%
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
#%%
# Load the datasets
miles_driven = pd.read_csv("miles-driven.csv",sep='|')
road_accidents = pd.read_csv("road-accidents.csv",sep='|')

# Display the first few rows of each dataset
print("Miles Driven Data:")
print(miles_driven.head())

print("\nRoad Accidents Data:")
print(road_accidents.head())
#%%
# Merge the datasets on the 'state' column
data = pd.merge(miles_driven, road_accidents, on="state")

# Display the merged dataset
print("Merged Data:")
print(data.head())

#%%
# Check for missing values
print("Missing Values:")
print(data.isnull().sum())

# Summary statistics
print("\nSummary Statistics:")
print(data.describe())

# Correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
#%%
# Select relevant features for clustering

features = data[["million_miles_annually", "drvr_fatl_col_bmiles", "perc_fatl_speed", "perc_fatl_1st_time"]]

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Convert scaled features back to a DataFrame
scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
print("Scaled Features:")
print(scaled_df.head())

#%%
# Use the Elbow Method to find the optimal number of clusters
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_df)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

#%%
# Choose the optimal number of clusters (e.g., 3 based on the Elbow Method)
kmeans = KMeans(n_clusters=5, random_state=42)
data["cluster"] = kmeans.fit_predict(scaled_df)

# Display the clusters
print("Data with Clusters:")
print(data.head())

#%%
# Reduce dimensionality using PCA for visualization
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_df)
data["pca_1"] = pca_features[:, 0]
data["pca_2"] = pca_features[:, 1]

# Plot the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x="pca_1", y="pca_2", hue="cluster", data=data, palette="viridis", s=100)
plt.title("Clusters of States")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.show()

#%%
# Group by clusters and calculate mean values
cluster_analysis = data.groupby("cluster").mean(numeric_only=True)
print("Cluster Analysis:")
print(cluster_analysis)

#%%
# Interpret the clusters
for cluster in sorted(data["cluster"].unique()):
    print(f"\nCluster {cluster}:")
    print(data[data["cluster"] == cluster]["state"].values)

#%%
# Suggest policy actions based on cluster characteristics
for cluster in sorted(data["cluster"].unique()):
    print(f"\nPolicy Suggestions for Cluster {cluster}:")
    if cluster == 0:
        print("Focus on improving road quality and reducing speeding.")
    elif cluster == 1:
        print("Implement stricter drunk driving laws and increase awareness campaigns.")
    elif cluster == 2:
        print("Invest in infrastructure to reduce fatal collisions and improve road safety.")
#%%
# Save the clustered data to a CSV file
data.to_csv("clustered_states.csv", index=False)
print("Clustered data saved to 'clustered_states.csv'.")
# %%
