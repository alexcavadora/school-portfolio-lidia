import os, time, json
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# === Linear Models ===
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron, PassiveAggressiveClassifier
)

# === SVM ===
from sklearn.svm import SVC, LinearSVC, NuSVC

# === Tree-Based Models ===
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
)

# === Naive Bayes ===
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, ComplementNB

# === Nearest Neighbors ===
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

# === Discriminant Analysis ===
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# === Neural Net ===
from sklearn.neural_network import MLPClassifier

# === External Models (if available) ===
try:
    from xgboost import XGBClassifier
    has_xgb = True
except ImportError:
    has_xgb = False

try:
    from lightgbm import LGBMClassifier
    has_lgbm = True
except ImportError:
    has_lgbm = False

try:
    from catboost import CatBoostClassifier
    has_catboost = True
except ImportError:
    has_catboost = False

RESULTS_FILE = "model_results.json"


# ----------------------------
# Utility functions
# ----------------------------
def load_small_dataset(prefix="./epsilon_small/train"):
    """Load dataset in dense or sparse format."""
    X_npy, y_npy = f"{prefix}_X.npy", f"{prefix}_y.npy"
    X_npz = f"{prefix}_X.npz"

    if os.path.exists(X_npy) and os.path.exists(y_npy):
        X = np.load(X_npy, allow_pickle=False).astype(np.float32, copy=False)
        y = np.load(y_npy, allow_pickle=False).astype(np.int8, copy=False)
        return X, y, False

    if os.path.exists(X_npz) and os.path.exists(y_npy):
        X = sparse.load_npz(X_npz).tocsr()
        y = np.load(y_npy, allow_pickle=False).astype(np.int8, copy=False)
        return X, y, True

    raise FileNotFoundError(f"Dataset not found under prefix {prefix}")


def metrics_dict(y_true, y_pred, y_score=None):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC_AUC": np.nan,
    }
    if y_score is not None:
        try:
            metrics["ROC_AUC"] = roc_auc_score(y_true, y_score)
        except Exception:
            pass
    return metrics


def evaluate_model(model, Xtr, Xte, y_train, y_test):
    """Train and evaluate one model."""
    t0 = time.time()
    model.fit(Xtr, y_train)
    t1 = time.time()

    y_pred = model.predict(Xte)
    y_score = None
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(Xte)[:, 1]
    elif hasattr(model, "decision_function"):
        s = model.decision_function(Xte)
        y_score = s if s.ndim == 1 else s[:, 1]

    metrics = metrics_dict(y_test, y_pred, y_score)
    metrics["TrainTime"] = t1 - t0
    return metrics


# ----------------------------
# Main benchmark loop
# ----------------------------
def main():
    # Parameters
    prefix = "./epsilon_small/train"
    test_size = 0.2
    seed = 1

    # Load dataset
    X, y, is_sparse = load_small_dataset(prefix)
    print(f"Loaded dataset: shape={X.shape}, sparse={is_sparse}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Scale features
    scaler = StandardScaler(with_mean=not is_sparse)
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    # Big zoo of models
    models = {
        # Linear
        "LogisticRegression": LogisticRegression(solver="saga", max_iter=200, n_jobs=-1, random_state=seed),
        "RidgeClassifier": RidgeClassifier(random_state=seed),
        "SGDClassifier": SGDClassifier(loss="log_loss", max_iter=1000, random_state=seed),
        "Perceptron": Perceptron(max_iter=1000, random_state=seed),
        "PassiveAggressive": PassiveAggressiveClassifier(max_iter=1000, random_state=seed),

        # SVMs
        "LinearSVC": LinearSVC(max_iter=2000, random_state=seed),
        "SVC_rbf": SVC(kernel="rbf", probability=True, random_state=seed),
        "NuSVC": NuSVC(probability=True, random_state=seed),

        # Trees / Ensembles
        "DecisionTree": DecisionTreeClassifier(random_state=seed),
        "ExtraTree": ExtraTreeClassifier(random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=seed),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=seed),
        "Bagging": BaggingClassifier(n_estimators=200, random_state=seed, n_jobs=-1),

        # Naive Bayes
        "GaussianNB": GaussianNB(),
        "BernoulliNB": BernoulliNB(),
        "MultinomialNB": MultinomialNB(),
        "ComplementNB": ComplementNB(),

        # Neighbors
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "NearestCentroid": NearestCentroid(),

        # Discriminant Analysis
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(),

        # Neural Net
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=seed),
    }

    if has_xgb:
        models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=200, random_state=seed)
    if has_lgbm:
        models["LightGBM"] = LGBMClassifier(n_estimators=200, random_state=seed)
    if has_catboost:
        models["CatBoost"] = CatBoostClassifier(n_estimators=200, verbose=0, random_state=seed)

    # Load previous results if exist
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            results = json.load(f)
    else:
        results = {}

    # Run multiple passes
    num_passes = 6  # keep small at first, can raise to 5+ overnight
    for it in range(num_passes):
        print(f"\n=== Iteration {it+1}/{num_passes} ===")
        for name, model in models.items():
            print(f"Training {name}...")
            try:
                metrics = evaluate_model(model, Xtr, Xte, y_train, y_test)
                print(f"{name} => F1={metrics['F1']:.4f}, Acc={metrics['Accuracy']:.4f}")
                if (name not in results) or (metrics["F1"] > results[name]["best_F1"]):
                    results[name] = {
                        "best_F1": metrics["F1"],
                        "best_metrics": metrics,
                        "params": model.get_params() if hasattr(model, "get_params") else {}
                    }
            except Exception as e:
                print(f"{name} failed: {e}")

        # Save progress after each iteration
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

    print("\n=== Final Best Models ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
