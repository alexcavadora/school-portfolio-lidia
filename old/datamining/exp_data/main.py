#### Loading the data
import numpy as np
import pandas as pd

df = pd.read_csv('data/Inc_Exp_Data.csv')

# Add a binary column for home ownership
df['Home_Ownership'] = df['Monthly EMI or Rent Amount'].apply(lambda x: 1 if x == 0 else 0)

print(df.head(5))

#### Main features of the data
print(df.mean(numeric_only=True))
print(df.median(numeric_only=True))
print(df.mode())
print(df.std(numeric_only=True))
print(df.var(numeric_only=True))
print(df.kurt(numeric_only=True))
print(df.skew(numeric_only=True))
for column in df:
    if df[column].dtype!=int:
        continue
    print(column, " min: ", df[column].min())
    print(column, " max: ", df[column].max())

import matplotlib.pyplot as plt
import seaborn as sns

# Plot distributions for numeric columns
numeric_columns = df.select_dtypes(include='number').columns

plt.figure(figsize=(15, 10))
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[column], kde=True, bins=20)
    plt.title(f'Distribution of {column}')
    plt.tight_layout()
plt.savefig('Distributions original.jpg')


# Countplot for the 'Highest_Qualified_Member' categorical column
plt.figure(figsize=(10, 8))
sns.countplot(data=df, x='Highest Education Level', palette='Set2')
plt.title('Distribution of Qualification Levels')
plt.xticks(rotation=45)
plt.savefig('Distribution of Qualification Levels original.jpg')


# Basic statistical summary
print("Basic Statistical Summary:")
print(df.describe())

# Correlation matrix
print("\nCorrelation Matrix:")
print(df.corr(numeric_only=True))

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Identify unique values for categorical columns
categorical_columns = df.select_dtypes(exclude='number').columns
for col in categorical_columns:
    print(f"\nUnique values in {col}:")
    print(df[col].value_counts())


#### Data transformations (preprocessing)
from sklearn.preprocessing import OrdinalEncoder

# Define the order of qualifications if there's a hierarchy
qualifications_order = ['Illiterate', 'Under-Graduate', 'Graduate', 'Post-Graduate', 'Professional']

# Apply ordinal encoding
ordinal_encoder = OrdinalEncoder(categories=[qualifications_order])
df['Highest Education Level'] = ordinal_encoder.fit_transform(df[['Highest Education Level']])

print(df[['Highest Education Level']].head(20))

# Fill missing values with median or a specific value depending on the column
df.fillna(df.median(numeric_only=True), inplace=True)

from sklearn.preprocessing import MinMaxScaler

# Select numerical columns for normalization, excluding 'Home_Ownership'
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
numeric_columns = [col for col in numeric_columns if col != 'Home_Ownership']

# Apply Min-Max scaling
scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

plt.figure(figsize=(15, 10))
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[column], kde=True, bins=20)
    plt.title(f'Distribution of {column}')
    plt.tight_layout()
plt.savefig('Distributions scaled.jpg')

# Calculate the correlation matrix
correlation_matrix = df.corr(numeric_only=True)

# Display the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)
# Set up the matplotlib figure
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)

# Set title and labels
plt.title('Correlation Matrix Heatmap', fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

# Show the plot

plt.savefig('Correlations scaled.jpg')
from sklearn.preprocessing import PowerTransformer

# Select numerical columns excluding the specified ones
numeric_columns_to_transform = [col for col in numeric_columns if col != 'Home_Ownership']

# Apply Yeo-Johnson transformation to correct skewness, excluding specific columns
pt = PowerTransformer(method='yeo-johnson', standardize=False)
df[numeric_columns_to_transform] = pt.fit_transform(df[numeric_columns_to_transform])

# Plot the transformed and normalized data
plt.figure(figsize=(15, 10))
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[column], kde=True, bins=20)
    plt.title(f'Transformed & Scaled Distribution of {column}')
    plt.tight_layout()
plt.savefig('Distributions scaled and normalized.jpg')    


#### Data mining

import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the correlation matrix
correlation_matrix = df.corr(numeric_only=True)

# Set up the matplotlib figure
plt.figure(figsize=(15, 10))

# Create a heatmap
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)

# Set title and labels
plt.title('Correlation Matrix Heatmap', fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

# Show the plot
plt.savefig('Correlation matrix scaled and normalized.jpg')


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Select relevant features for clustering
features = df[['Monthly Household Income', 'Monthly Household Expense', 'Number of family members', 
               'Monthly EMI or Rent Amount', 'Annual Household Income', 
               'Number of Earning Members']]

# Fit the K-Means model
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(features)

# Inverse transform to get original values for better readability
df[numeric_columns_to_transform] = pt.inverse_transform(df[numeric_columns_to_transform])
df[numeric_columns_to_transform] = scaler.inverse_transform(df[numeric_columns_to_transform])


# Visualize the clusters,
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='Highest Education Level', y='Annual Household Income', hue='Cluster', palette='viridis', style='Cluster', s=100)
plt.title('Clusters based on Highest Education Level and Income')
plt.xlabel('Highest Education Level')
plt.ylabel('Annual Household Income')
plt.legend(title='Cluster')
plt.savefig('Clustered returned to original.jpg')



# Grouping by cluster and calculating mean for each feature
cluster_means = df.groupby('Cluster').mean()

# Display the mean values of each feature for each cluster
print("Mean Feature Values by Cluster:")
print(cluster_means)

# Analyzing the number of family members by cluster
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Cluster', hue='Number of family members')
plt.title('Number of Family Members by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.legend(title='Number of Family Members')
plt.tight_layout()
plt.savefig('Clustered comparisons2.jpg')

# Grouping by cluster and calculating average income for each cluster
cluster_income = df.groupby('Cluster')['Monthly Household Income'].mean()

# Calculate overall mode for the number of family members and monthly income
overall_mode_family_members = df['Number of family members'].mode()[0]
overall_mode_income = df['Monthly Household Income'].mode()[0]

# Display the results
print("Average Monthly Household Income for each Cluster:")
print(cluster_income)

print(f"\nOverall mode for number of family members: {overall_mode_family_members}")
print(f"Overall mode for Monthly Household Income: {overall_mode_income}")

# Visualization
plt.figure(figsize=(10, 6))

# Plot average income per cluster
sns.barplot(x=cluster_income.index, y=cluster_income.values, palette='Set3')
plt.title('Average Monthly Household Income by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average Monthly Household Income')
plt.tight_layout()
plt.savefig('Average Monthly Household Income by Cluster.png')

# Calculate average income per family member for each cluster
df['Income per Family Member'] = df['Monthly Household Income'] / df['Number of family members']

# Grouping by cluster and calculating average income per family member for each cluster
cluster_income_per_member = df.groupby('Cluster')['Income per Family Member'].mean()
print('mode')
for category in df[numeric_columns]:
    print(category, df[category].mode()[0])

# Display the results
print("Average Income per Family Member for each Cluster:")
print(cluster_income_per_member)