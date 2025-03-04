import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import time

# Load the dataset
file_path = "features_10.csv"  # Update the file path if necessary
data = pd.read_csv(file_path)

# Encode the categorical target variable
label_encoder = LabelEncoder()
data['category'] = label_encoder.fit_transform(data['category'])

# Shuffle the dataset
data = shuffle(data, random_state=1)

# Split features and target
X = data.iloc[:, :-1].values  # Use all features for training
y = data.iloc[:, -1].values

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

# Apply LDA
lda = LDA(n_components=len(np.unique(y)) - 1) 
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

#Define kernel types and their names
kernels = {
    'linear': 'Linear SVM',
    'poly_2': 'Quadratic SVM',
    'poly_3': 'Cubic SVM',
    'poly_5': 'Fifth-degree SVM',
    'rbf': 'RBF SVM'
}

# Loop through each kernel type
for kernel, name in kernels.items():
    if kernel == 'rbf':
        clf = SVC(kernel='rbf', random_state=1, degree=1) 
    else:
        degree = int(kernel.split('_')[-1]) if 'poly' in kernel else 1
        clf = SVC(
            kernel='poly' if 'poly' in kernel else 'linear',
            degree=degree,
            random_state=1,
            C=0.85,
            gamma='scale'
        )

    # Compute learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        clf,
        np.concatenate((X_train, X_test)),
        np.concatenate((y_train, y_test)),
        cv=5,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        shuffle=True,  # Shuffle data at each split
        random_state=1
    )

    # Calculate mean and standard deviation for training scores
    train_scores_mean = train_scores.mean(axis=1)
    train_scores_std = train_scores.std(axis=1)

    # Calculate mean and standard deviation for test scores
    test_scores_mean = test_scores.mean(axis=1)
    test_scores_std = test_scores.std(axis=1)

    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.title(f"Learning Curve - {name}")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.ylim(0, 1.1)
    plt.grid()

    # Plot the training scores
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="darkred",
        hatch='\\\\\\'
    )
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")

    # Plot the validation scores
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="darkgreen",
        hatch='///'
    )
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    # Calculate train split point (80% of max training size)
    train_split_x = 800 * 0.8
    
    # Find the index of the train size closest to train_split_x
    train_split_index = np.argmin(np.abs(train_sizes - train_split_x))
    
    # Get the cross-validation score at the train split point
    train_split_y = test_scores_mean[train_split_index]

    # Add vertical line at train split
    plt.axvline(x=train_split_x, ls='--', color='blue', label='Train split')

    # Add intersection point
    plt.scatter(
        train_split_x, 
        train_split_y, 
        color='lightgreen',
        marker='*',
        edgecolor='darkred',
        s=200,  # marker size
        zorder=5  # ensure point is on top of other elements
    )
    plt.annotate(
        f'Accuracy={train_split_y:.2f}', 
        (train_split_x, train_split_y),
        xytext=(20, 20),  # offset text slightly
        textcoords='offset points',
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", ec="b", alpha=0.3),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5')
    )

    plt.legend(loc="best")
    plt.show()

    # Rest of the code remains the same (training, evaluation, etc.)
    # Fit the classifier on the full training set
    start_time = time.time()
    clf.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Predict on the test set
    start_time = time.time()
    y_pred = clf.predict(X_test)
    testing_time = time.time() - start_time

    # Evaluate and print results
    print(f"Results for {name}:")
    print(f"Training Time: {training_time:.4f} seconds")
    print(f"Testing Time: {testing_time:.4f} seconds")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {name}')
    plt.xticks(rotation=90)
    plt.show()