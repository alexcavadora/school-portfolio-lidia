#%%
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
from operator import itemgetter

#%%
# Step 1: Inspect the file
# Read and print the first few rows
data = pd.read_csv("transfusion.data")
print(data.head())
#%%
# Step 2: Load the blood donation data
# No extra steps needed as it's loaded in Step 1
#%%
# Step 3: Inspect the data structure
print(data.info())
#%%
# Step 4: Rename the target column
data.rename(columns={"whether he/she donated blood in March 2007": "target"}, inplace=True)
print(data.head(2))
#%%
# Step 5: Check target incidence
print(data["target"].value_counts(normalize=True).round(3))
#%%
# Step 6: Split the data into original and reserve sets
X = data.drop("target", axis=1)
y = data["target"]
X_original, X_reserve, y_original, y_reserve = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(X_original.head(2))
#%%
# Step 7: Split the original dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_original, y_original, test_size=0.25, stratify=y_original, random_state=42)
print(X_train.head(2))
#%%
# Step 8: Use TPOT to find the best model
tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, scoring='roc_auc', random_state=42, disable_update_check=True, config_dict='TPOT light')
tpot.fit(X_train, y_train)
tpot_auc_score = roc_auc_score(y_test, tpot.predict_proba(X_test)[:, 1])
print(f"TPOT AUC score: {tpot_auc_score:.4f}")
print("Best pipeline is:")
for idx, (name, transform) in enumerate(tpot.fitted_pipeline_.steps, start=1):
    print(f"Step {idx}: {name} - {transform}")
#%%
# Step 9: Check variance and log normalization
print(X_train.var().round(3))
col_to_normalize = X_train.var().idxmax()
X_train_normed = X_train.copy()
X_test_normed = X_test.copy()
X_train_normed[col_to_normalize] = np.log(X_train[col_to_normalize])
X_test_normed[col_to_normalize] = np.log(X_test[col_to_normalize])
X_train_normed.drop(columns=[col_to_normalize], inplace=True)
X_test_normed.drop(columns=[col_to_normalize], inplace=True)
print(X_train_normed.var().round(3))
#%%
# Step 10: Train logistic regression model
logreg = LogisticRegression(solver='sag', max_iter=100, random_state=42)
logreg.fit(X_train_normed, y_train)
logreg_auc_score = roc_auc_score(y_test, logreg.predict_proba(X_test_normed)[:, 1])
print(f"Logistic Regression AUC score: {logreg_auc_score:.4f}")
#%%
# Step 11: Tune logistic regression hyperparameters
param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'fit_intercept': [True, False],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter': [100, 1000, 2500, 5000]
}
grid_search = GridSearchCV(LogisticRegression(solver='sag', random_state=42), param_grid, scoring='roc_auc', cv=5, verbose=True, n_jobs=-1)
grid_search.fit(X_train_normed, y_train)
logreg_cv_best_score = grid_search.best_score_
print(f"Best GridSearch AUC score: {logreg_cv_best_score:.4f}")
#%%
# Step 12: Update the model and normalize for the reserve data
X_original_normed = X_original.copy()
X_reserve_normed = X_reserve.copy()
X_original_normed[col_to_normalize] = np.log(X_original[col_to_normalize])
X_reserve_normed[col_to_normalize] = np.log(X_reserve[col_to_normalize])
X_original_normed.drop(columns=[col_to_normalize], inplace=True)
X_reserve_normed.drop(columns=[col_to_normalize], inplace=True)
print(X_reserve_normed.var().round(3))
#%%
# Step 13: Validate the final model with reserve data
logreg_reserva = LogisticRegression(solver='sag', max_iter=100, random_state=42)
logreg_reserva.fit(X_reserve_normed, y_reserve)
logreg_reserva_auc_score = roc_auc_score(y_reserve, logreg_reserva.predict_proba(X_reserve_normed)[:, 1])
print(f"Final AUC score on reserve data: {logreg_reserva_auc_score:.4f}")
#%%
# Step 14: Sort models by AUC
models = [
    ("TPOT", tpot_auc_score),
    ("Logistic Regression", logreg_auc_score),
    ("Logistic Regression GridSearch", logreg_cv_best_score),
    ("Logistic Regression Reserve", logreg_reserva_auc_score)
]
sorted_models = sorted(models, key=itemgetter(1), reverse=True)
print("Models sorted by AUC:")
for model in sorted_models:
    print(f"{model[0]}: {model[1]:.4f}")
