import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Load a simple dataset
# Let's use the Iris dataset (a multi-class dataset with 3 classes)
iris = datasets.load_iris()
X = iris.data[:, :2]  # Only take the first two features for visualization
y = iris.target

# 2. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Standardize the data (helps with SVM performance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Train SVM with polynomial kernels of varying degrees and perform cross-validation
degrees = range(10)  # Polynomial degrees to test
cv_scores = []  # Store cross-validation scores for each degree

for d in degrees:
    svm_poly = SVC(kernel='poly', degree=d, C=1.0)
    scores = cross_val_score(svm_poly, X_train, y_train, cv=5)  # 5-fold cross-validation
    cv_scores.append(np.mean(scores))
    print(f"Degree {d}: Cross-validation Accuracy = {np.mean(scores):.4f}")

# 5. Plot Cross-validation accuracy vs. degree of the polynomial kernel
plt.figure(figsize=(8, 6))
plt.plot(degrees, cv_scores, marker='o', linestyle='--', color='b')
plt.title('Cross-validation Accuracy vs Degree of Polynomial Kernel')
plt.xlabel('Degree of Polynomial Kernel (d)')
plt.ylabel('Cross-validation Accuracy')
plt.grid(True)
plt.show()

# 6. Choose the best degree based on cross-validation and train on the full training set
best_degree = degrees[np.argmax(cv_scores)]
print(f"Best Degree from Cross-validation: {best_degree}")

# 7. Train the SVM with the best degree
best_svm_poly = SVC(kernel='poly', degree=best_degree, C=1.0)
best_svm_poly.fit(X_train, y_train)

# 8. Evaluate on the test set
y_pred = best_svm_poly.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy with Polynomial Kernel (degree={best_degree}): {test_accuracy:.4f}")

# 9. Visualize the decision boundary for the best polynomial kernel
def plot_decision_boundary(model, X, y):
    h = .02  # Mesh step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title(f"Decision Boundary (Degree={best_degree})")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

plot_decision_boundary(best_svm_poly, X_test, y_test)

svm_rbf = SVC(kernel='rbf', C=4.0, gamma='scale')  # Use default gamma
svm_rbf.fit(X_train, y_train)

# Predict and evaluate
y_pred_rbf = svm_rbf.predict(X_test)
test_accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
print(f"Test Accuracy with RBF Kernel: {test_accuracy_rbf:.4f}")

# Define a function to plot decision boundaries
def plot_decision_boundary(svm, X, y, title):
    # Create a grid to plot decision boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Predict on the grid
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=100, cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.grid()
    plt.colorbar()

# Plot the RBF kernel decision boundary
plot_decision_boundary(svm_rbf, X, y, title='SVM with RBF Kernel Decision Boundary')

# Show the plot
plt.show()
