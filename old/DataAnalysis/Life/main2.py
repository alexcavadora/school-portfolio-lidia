# 2. Cargando los datos de las donaciones de sangre
import pandas as pd

transfusion = pd.read_csv('transfusion.data')
transfusion.head()

# 3. Inspección de la base de datos de transfusión
transfusion.info()

# 4. Creando la columna objetivo
transfusion = transfusion.rename(columns={'whether he/she donated blood in March 2007': 'target'})
transfusion.head(2)

# 5. Comprobación de la incidencia objetivo
transfusion.target.value_counts(normalize=True).round(3)

# 6. Dividir el conjuntos de datos transfusión en original y reserva
from sklearn.model_selection import train_test_split

X_original, X_reserva, y_original, y_reserva = train_test_split(
    transfusion.drop('target', axis=1),
    transfusion.target,
    stratify=transfusion.target,
    test_size=0.2,
    random_state=42
)

X_original.head(2)

# 7. Dividir el conjuntos de datos originales de transfusión en entrenamiento y prueba
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_original,
    y_original,
    stratify=y_original,
    test_size=0.25,
    random_state=42
)

X_train.head(2)

# 7. Selección del modelo mediante TPOT
from tpot import TPOTClassifier
from sklearn.metrics import roc_auc_score

tpot = TPOTClassifier(
    generations=5,
    population_size=20,
    verbose=2,
    scorers='roc_auc',
    random_state=42,
    config_dict='TPOT light'
)

tpot.fit(X_train, y_train)

tpot_auc_score = roc_auc_score(y_test, tpot.predict_proba(X_test)[:, 1])
print(round(tpot_auc_score, 4))

print("el mejor pipeline es:")
for idx, (name, transform) in enumerate(tpot.fitted_pipeline_.steps, start=1):
    print(idx, transform)

# 8. Verificando la varianza
X_train.var().round(3)

# 9. Log normalización de registros
import numpy as np

X_train_normed = X_train.copy()
X_test_normed = X_test.copy()

col_to_normalize = 'Monetary (c.c. blood)'

for df_ in [X_train_normed, X_test_normed]:
    df_['monetary_log'] = np.log(df_[col_to_normalize])
    df_.drop(col_to_normalize, inplace=True)

X_train_normed.var().round(3)

# 10. Entrenando el modelo de regresión logística
from sklearn import linear_model

logreg = linear_model.LogisticRegression(solver='sag', max_iter=100, random_state=42)
logreg.fit(X_train_normed, y_train)

logreg_auc_score = roc_auc_score(y_test, logreg.predict_proba(X_test_normed)[:, 1])
print(f"AUC score: {round(logreg_auc_score, 4)}")

# 11. Tunear los hiperparámetros de la regresión logística
from sklearn.model_selection import GridSearchCV

param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'fit_intercept': [True, False],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter': [100, 1000, 2500, 5000]
}

logreg_cv = GridSearchCV(
    linear_model.LogisticRegression(random_state=42),
    param_grid,
    scoring='roc_auc',
    cv=5,
    verbose=True,
    n_jobs=-1
)

logreg_cv.fit(X_train_normed, y_train)

print(f"Mejores parámetros: {logreg_cv.best_params_}")
print(f"Mejor AUC: {logreg_cv.best_score_}")

# 12. Actualizar los hiperparámetros estimados y normalizar
X_original_normed = X_original.copy()
X_reserva_normed = X_reserva.copy()

col_to_normalize = 'Monetary (c.c. blood)'

for df_ in [X_original_normed, X_reserva_normed]:
    df_['monetary_log'] = np.log(df_[col_to_normalize])
    df_.drop(col_to_normalize, inplace=True)

X_reserva_normed.var().round(3)

# 12. Validación final con los datos de reserva
logreg_reserva = linear_model.LogisticRegression(**logreg_cv.best_params_, random_state=42)
logreg_reserva.fit(X_original_normed, y_original)

logreg_reserva_auc_score = roc_auc_score(y_reserva, logreg_reserva.predict_proba(X_reserva_normed)[:, 1])
print(f"AUC score: {round(logreg_reserva_auc_score, 4)}")

# 13. Conclusión
from operator import itemgetter

models = [
    ('TPOT', tpot_auc_score),
    ('Regresión Logística', logreg_auc_score),
    ('Regresión Logística con CV', logreg_cv.best_score_),
    ('Regresión Logística Final', logreg_reserva_auc_score)
]

sorted_models = sorted(models, key=itemgetter(1), reverse=True)
print(sorted_models)