#%%
# Step 1: Load and inspect the dataset
import pandas as pd

# Load the dataset
cc_apps = pd.read_csv("cc_approvals.data", header=None)

# Display the first 5 rows
print(cc_apps.head())

#%%
# Step 2: Inspect the dataset structure
# Summary statistics for numerical columns
print(cc_apps.describe())

# Information about the DataFrame
print(cc_apps.info())

# Display the last 17 rows to check for missing values
print(cc_apps.tail(17))

#%%
# Step 3: Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Drop columns 11 and 13 (irrelevant features)
cc_apps = cc_apps.drop([11, 13], axis=1)

# Split the data into training and testing sets
cc_apps_train, cc_apps_test = train_test_split(cc_apps, test_size=0.33, random_state=42)

# Display the shapes of the resulting datasets
print(cc_apps_train.shape)
print(cc_apps_test.shape)

#%%
# Step 4: Handle missing values (Part 1)
import numpy as np

# Replace '?' with NaN
cc_apps_train = cc_apps_train.replace('?', np.nan)
cc_apps_test = cc_apps_test.replace('?', np.nan)

# Display the first few rows to verify
print(cc_apps_train.head())
print(cc_apps_test.head())

#%%
# Step 5: Handle missing values (Part 2)
# Impute missing numerical values with the mean
for col in cc_apps_train.columns:
    if cc_apps_train[col].dtype in ['float64', 'int64']:
        mean_value = cc_apps_train[col].mean()
        cc_apps_train[col].fillna(mean_value, inplace=True)
        cc_apps_test[col].fillna(mean_value, inplace=True)

# Check if there are any remaining NaNs in numerical columns
print(cc_apps_train.isna().sum())
print(cc_apps_test.isna().sum())

#%%
# Step 6: Handle missing values (Part 3)
# Impute missing categorical values with the most frequent value
for col in cc_apps_train.columns:
    if cc_apps_train[col].dtype == 'object':
        most_frequent_value = cc_apps_train[col].value_counts().idxmax()
        cc_apps_train[col].fillna(most_frequent_value, inplace=True)
        cc_apps_test[col].fillna(most_frequent_value, inplace=True)

# Check if there are any remaining NaNs
print(cc_apps_train.isna().sum())
print(cc_apps_test.isna().sum())

#%%
# Step 7: Preprocess the data (Part 1)
# Convert categorical data into numerical data using one-hot encoding
cc_apps_train = pd.get_dummies(cc_apps_train)
cc_apps_test = pd.get_dummies(cc_apps_test)

# Align the columns of the test set with the training set
cc_apps_test = cc_apps_test.reindex(columns=cc_apps_train.columns, fill_value=0)

# Display the first few rows to verify
print(cc_apps_train.head())
print(cc_apps_test.head())

#%%
# Step 8: Preprocess the data (Part 2)

# Separate features and labels
X_train = cc_apps_train.iloc[:, :-1]
y_train = cc_apps_train.iloc[:, -1]
X_test = cc_apps_test.iloc[:, :-1]
y_test = cc_apps_test.iloc[:, -1]
from sklearn.preprocessing import MinMaxScaler

# Check column names before renaming
print("X_train columns before renaming:", X_train.columns)
print("X_test columns before renaming:", X_test.columns)

# Add a prefix to all column names to make them non-numeric
X_train.columns = ['col_' + str(col) for col in X_train.columns]
X_test.columns = ['col_' + str(col) for col in X_test.columns]

# Verify column names after renaming
print("X_train columns after renaming:", X_train.columns)
print("X_test columns after renaming:", X_test.columns)



# Scale the features to a range of 0 to 1
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.transform(X_test)

# Display the scaled data
print("Scaled Training Data:")
print(rescaledX_train[:5])
print("\nScaled Testing Data:")
print(rescaledX_test[:5])
#%%
# Step 9: Fit a logistic regression model
from sklearn.linear_model import LogisticRegression

# Create a logistic regression model
logreg = LogisticRegression()

# Fit the model to the training data
logreg.fit(rescaledX_train, y_train)

#%%
# Step 10: Make predictions and evaluate performance
from sklearn.metrics import confusion_matrix, accuracy_score

# Make predictions on the test set
y_pred = logreg.predict(rescaledX_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

#%%
# Step 11: Grid search for hyperparameter tuning
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'tol': [0.01, 0.001, 0.0001],
    'max_iter': [100, 150, 200]
}

# Create a GridSearchCV object
grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)

# Fit the grid search to the training data
grid_model.fit(rescaledX_train, y_train)

# Display the best parameters and score
print(f"Best Parameters: {grid_model.best_params_}")
print(f"Best Score: {grid_model.best_score_:.4f}")

#%%
# Step 12: Evaluate the best model
# Extract the best model
best_model = grid_model.best_estimator_

# Make predictions with the best model
y_pred_best = best_model.predict(rescaledX_test)

# Calculate accuracy
best_accuracy = accuracy_score(y_test, y_pred_best)
print(f"Best Model Accuracy: {best_accuracy:.4f}")

# Display the confusion matrix for the best model
best_conf_matrix = confusion_matrix(y_test, y_pred_best)
print("Best Model Confusion Matrix:")
print(best_conf_matrix)
# %%
