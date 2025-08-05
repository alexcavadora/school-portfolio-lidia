from ucimlrepo import fetch_ucirepo
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd

# Fetch dataset
mushroom = fetch_ucirepo(id=73)

# Data (as pandas dataframes)
X = mushroom.data.features
y = mushroom.data.targets

# Encode target variable (since y is categorical)
le = LabelEncoder()
y = le.fit_transform(y.values.ravel())  # Flatten in case it's a DataFrame

# Encode categorical features
X_encoded = pd.get_dummies(X)  # One-hot encoding for categorical features

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, shuffle=True, test_size=0.2, random_state=42)

# Train SVM model
model = LinearSVC(max_iter=10000, dual=False)  # Setting dual=False for small datasets
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Metadata
#print(mushroom.metadata)

# Variable information
#print(mushroom.variables)
