import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
file_path = "features.csv"  # Adjust path if necessary
data = pd.read_csv(file_path)

# Encode the categorical target variable
label_encoder = LabelEncoder()
data['category'] = label_encoder.fit_transform(data['category'])

# Split features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Define kernel types and their names
kernels = {
    'linear': 'Linear SVM',
    'poly_2': 'Quadratic SVM (Degree 2)',
    'poly_3': 'Cubic SVM (Degree 3)',
    'poly_5': 'Fifth-degree SVM (Degree 5)'
}

# Train and evaluate each SVM
svm_results = {}
for kernel, name in kernels.items():
    degree = int(kernel.split('_')[-1]) if 'poly' in kernel else 1
    clf = SVC(kernel='poly' if 'poly' in kernel else 'linear', degree=degree, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    svm_results[name] = {
        'model': clf,
        'report': classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    }

# Plot decision boundaries for the first two features
def plot_decision_boundary(clf, X, y, title):
    h = 0.02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict the results
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Visualize the decision boundaries (first two features)
for kernel, name in kernels.items():
    clf = svm_results[name]['model']
    plot_decision_boundary(clf, X_train[:, :2], y_train, title=f"{name} Decision Boundary")

# Print classification reports
for name, result in svm_results.items():
    print(f"Classification Report for {name}:")
    print(result['report'])

