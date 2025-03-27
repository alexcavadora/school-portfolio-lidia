#%%
# Step 1: Read data
import pandas as pd
transfusion = pd.read_csv("transfusion.data")
print(transfusion.head())

#%%
# Step 2: Data info
print(transfusion.info())

#%%
# Step 3: Rename target column
transfusion.rename(columns={'whether he/she donated blood in March 2007': 'target'}, inplace=True)
print(transfusion.head(2))

#%%
# Step 4: Target distribution
print(transfusion['target'].value_counts(normalize=True).round(3))

#%%
# Step 5: Initial split
from sklearn.model_selection import train_test_split

X = transfusion.drop('target', axis=1)
y = transfusion['target']

# Split into temp and final test set
X_temp, X_reserva, y_temp, y_reserva = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

#%%
# Step 6: Train-validation split
X_train, X_test, y_train, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)

#%%
# Step 7: TPOT setup (updated parameters)
#import os
#os.environ["TPOT_NO_DASK"] = "true"
from tpot import TPOTClassifier
from sklearn.metrics import roc_auc_score

# Updated TPOT configuration
tpot = TPOTClassifier(
    generations=5,
    population_size=20,
    cv=5,
    scorers=['roc_auc'],  # Correct parameter name for scoring
    scorers_weights=[1.0],  # Weight for the scorer
    verbose=2,
    random_state=42,
    n_jobs=-1
)

tpot.fit(X_train, y_train)

# Evaluation
tpot_auc = roc_auc_score(y_test, tpot.predict_proba(X_test)[:, 1])
print(f"TPOT AUC: {tpot_auc:.4f}")

#%%
# Step 8: Feature engineering
import numpy as np

# Log-transform high-variance feature
col_to_normalize = X_train.var().idxmax()
for df in [X_train, X_test, X_temp, X_reserva]:
    df['monetary_log'] = np.log(df[col_to_normalize])
    df.drop(col_to_normalize, axis=1, inplace=True)

#%%
# Step 9: Logistic regression baseline
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver='saga', max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)
logreg_auc = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])
print(f"Baseline AUC: {logreg_auc:.4f}")

#%%
# Step 10: Hyperparameter tuning (fixed data usage)
from sklearn.model_selection import GridSearchCV

# Updated parameter grid with valid combinations
param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': [0.01, 0.1, 1, 10],
    'solver': ['saga'],  # Only solver supporting all penalties
    'l1_ratio': [0.3, 0.5, 0.7]
}

logreg_cv = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1
)

# Corrected to use training data
logreg_cv.fit(X_train, y_train)

print(f"Best params: {logreg_cv.best_params_}")
best_auc = roc_auc_score(y_test, logreg_cv.predict_proba(X_test)[:, 1])
print(f"Tuned AUC: {best_auc:.4f}")

#%%
# Step 11: Final evaluation on holdout set
final_model = logreg_cv.best_estimator_
final_model.fit(X_temp, y_temp)  # Train on full temp data
reserva_auc = roc_auc_score(y_reserva, final_model.predict_proba(X_reserva)[:, 1])
print(f"Final Holdout AUC: {reserva_auc:.4f}")
# %%
