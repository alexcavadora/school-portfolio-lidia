import numpy as np
import pandas as pd
import argparse
import os
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def calculate_statistical_features(sum_hist, diff_hist):
    mean_s = np.mean(sum_hist)
    mean_d = np.mean(diff_hist)

    variance_s = np.var(sum_hist)
    variance_d = np.var(diff_hist)

    correlation = np.sum((sum_hist - mean_s)**2) - np.sum((diff_hist)**2)

    contrast = np.sum(diff_hist**2)

    homogeneity = np.sum(1 / (1 + diff_hist))

    shadow_cluster = np.sum((sum_hist - mean_s)**3)

    prominence_cluster = np.sum((sum_hist - mean_s)**4)

    features = [mean_s, variance_s, correlation, contrast, homogeneity, shadow_cluster, prominence_cluster]
    return np.array(features)

def compute_histograms(signals, window_size=100):
    n_samples, n_phases = signals.shape
    n_windows = n_samples - window_size + 1
    histogram_features = []

    for phase in range(n_phases):
        phase_signals = signals[:, phase]

        sum_hist = []
        diff_hist = []

        for i in range(n_windows):
            window = phase_signals[i:i+window_size]
            sum_val = window[0] + window[-1]
            diff_val = window[-1] - window[0]

            sum_hist.append(sum_val)
            diff_hist.append(diff_val)

        # Convert histograms to numpy arrays and compute statistical properties
        sum_hist = np.array(sum_hist)
        diff_hist = np.array(diff_hist)

        stats_features = calculate_statistical_features(sum_hist, diff_hist)
        histogram_features.append(stats_features)

    return np.concatenate(histogram_features)

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

pathBD = os.path.join(os.getcwd(), args["dataset"])

subfolders = [os.path.join(pathBD, name) for root, dirs, files in os.walk(pathBD) for name in dirs]
nclasses = len(subfolders)

n_samples = 5000
n_phases = 3
n_rep = 5
window_size = 13
Data = np.ndarray(shape=(nclasses, n_rep, n_samples, n_phases))

for n, classe in enumerate(subfolders):
    path = subfolders[n]
    files = [os.path.join(path, fname) for root, dirs, files in os.walk(path) for fname in files if fname.endswith((".csv"))]
    nfiles = len(files)

    for f in range(nfiles):
        df = pd.read_csv(files[f], header=None)
        Signals = df.to_numpy()
        Data[n, f, :, :] = Signals

print("[INFO] Data shape = {}".format(Data.shape))

features = []
labels = []

for n in range(nclasses):
    for rep in range(n_rep):
        signals = Data[n, rep, :, :]
        # Compute histogram features for the current signal
        hist_features = compute_histograms(signals, window_size=window_size)
        features.append(hist_features)
        labels.append(n)

X = np.array(features)
y = np.array(labels)
print("[INFO] Features shape = {}".format(X.shape))

# Scaling the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA (optional step, you can skip it if unnecessary)
pca = PCA(n_components=8)
X_pca = pca.fit_transform(X_scaled)

# LDA (optional step, you can skip it if unnecessary)
lda = LDA(n_components=min(8, nclasses - 1))
X_lda = lda.fit_transform(X_scaled, y)

# X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=2)
#X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=0.2, random_state=2)

# Use X_lda or X_pca depending on what you want as input features
X = X_lda  # or X_pca if you prefer
#-----------------------------------------------
# Cross-validation with Random Forest
#-----------------------------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=12)
y_pred_cv_rf = cross_val_predict(rf, X, y, cv=5)
rf_cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
print(f"[INFO] Random Forest Cross-Validation Accuracy (5 folds): {np.mean(rf_cv_scores) * 100:.2f}%")
print("[INFO] Random Forest Confusion Matrix (Cross-Validation):\n", confusion_matrix(y, y_pred_cv_rf))

#-----------------------------------------------
# Cross-validation with Decision Tree
#-----------------------------------------------
dt = DecisionTreeClassifier(random_state=12)
y_pred_cv_dt = cross_val_predict(dt, X, y, cv=5)
dt_cv_scores = cross_val_score(dt, X, y, cv=5, scoring='accuracy')
print(f"[INFO] Decision Tree Cross-Validation Accuracy (5 folds): {np.mean(dt_cv_scores) * 100:.2f}%")
print("[INFO] Decision Tree Confusion Matrix (Cross-Validation):\n", confusion_matrix(y, y_pred_cv_dt))

#-----------------------------------------------
# Cross-validation with KNN
#-----------------------------------------------
knn = KNeighborsClassifier(n_neighbors=4)
y_pred_cv_knn = cross_val_predict(knn, X, y, cv=5)
knn_cv_scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
print(f"[INFO] KNN Cross-Validation Accuracy (5 folds): {np.mean(knn_cv_scores) * 100:.2f}%")
print("[INFO] KNN Confusion Matrix (Cross-Validation):\n", confusion_matrix(y, y_pred_cv_knn))
