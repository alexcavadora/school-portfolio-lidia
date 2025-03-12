#%%
# Paso 1: Leer el archivo
import pandas as pd

# Cargar el archivo transfusion.data
transfusion = pd.read_csv("transfusion.data")

# Mostrar las primeras 5 líneas del archivo
print(transfusion.head())
#%%
# Imprimir un resumen conciso del DataFrame
print(transfusion.info())
#%%
# Renombrar la columna 'whether he/she donated blood in March 2007' a 'target'
transfusion.rename(columns={'whether he/she donated blood in March 2007': 'target'}, inplace=True)

# Verificar el cambio
print(transfusion.head(2))
#%%
# Imprimir la incidencia objetivo
print(transfusion['target'].value_counts(normalize=True).round(3))
#%%
from sklearn.model_selection import train_test_split

# Dividir el conjunto de datos en original y reserva
X_original, X_reserva, y_original, y_reserva = train_test_split(transfusion.drop('target', axis=1), transfusion['target'], test_size=0.2, random_state=42, stratify=transfusion['target'])

# Verificar la división
print(X_original.head(2))

#%%
# Dividir el conjunto de datos original en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_original, y_original, test_size=0.25, random_state=42, stratify=y_original)

# Verificar la división
print(X_train.head(2))

#%%
from tpot import TPOTClassifier
from sklearn.metrics import roc_auc_score

# Crear una instancia de TPOTClassifier
tpot = TPOTClassifier(generations=5, population_size=20, verbose=2, scorers='roc_auc', random_state=42)

# Ajustar el modelo
tpot.fit(X_train, y_train)

# Estimar el área bajo la curva ROC (AUC)
tpot_auc_score = roc_auc_score(y_test, tpot.predict_proba(X_test)[:, 1])
print(f"AUC score: {tpot_auc_score:.4f}")

# Mostrar el mejor pipeline
print("El mejor pipeline es:")
for idx, (name, transform) in enumerate(tpot.fitted_pipeline_.steps, start=1):
    print(f"Paso {idx}: {transform}")
#%%
# Imprimir la varianza de X_train
print(X_train.var().round(3))
#%%
import numpy as np

# Copiar los datos de entrenamiento y prueba
X_train_normed = X_train.copy()
X_test_normed = X_test.copy()

# Identificar la columna con la mayor varianza
col_to_normalize = X_train.var().idxmax()

# Log-normalizar la columna con mayor varianza
for df in [X_train_normed, X_test_normed]:
    df['monetary_log'] = np.log(df[col_to_normalize])
    df.drop(col_to_normalize, axis=1, inplace=True)

# Verificar la varianza después de la normalización
print(X_train_normed.var().round(3))
#%%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Crear una instancia de LogisticRegression
logreg = LogisticRegression(solver='sag', max_iter=100, random_state=42)

# Entrenar el modelo
logreg.fit(X_train_normed, y_train)

# Estimar el área bajo la curva ROC (AUC)
logreg_auc_score = roc_auc_score(y_test, logreg.predict_proba(X_test_normed)[:, 1])
print(f"AUC score: {logreg_auc_score:.4f}")
#%%
from sklearn.model_selection import GridSearchCV

# Configurar la búsqueda de cuadrícula
param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'fit_intercept': [True, False],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter': [100, 1000, 2500, 5000]
}

# Crear la instancia de GridSearchCV
logreg_cv = GridSearchCV(LogisticRegression(), param_grid, scoring='roc_auc', cv=5, verbose=True, n_jobs=-1)

# Ajustar el modelo
logreg_cv.fit(X_test_normed, y_test)

# Imprimir el mejor parámetro y AUC calculado
print(f"Mejores parámetros: {logreg_cv.best_params_}")
print(f"AUC score: {logreg_cv.best_score_:.4f}")
#%%
# Copiar los datos originales y de reserva
X_original_normed = X_original.copy()
X_reserva_normed = X_reserva.copy()

# Log-normalizar la columna con mayor varianza
for df in [X_original_normed, X_reserva_normed]:
    df['monetary_log'] = np.log(df[col_to_normalize])
    df.drop(col_to_normalize, axis=1, inplace=True)

# Verificar la varianza después de la normalización
print(X_reserva_normed.var().round(3))
#%%
# Crear una instancia de LogisticRegression con los mejores hiperparámetros
logreg_reserva = LogisticRegression(**logreg_cv.best_params_)

# Ajustar el modelo con los datos de reserva
logreg_reserva.fit(X_reserva_normed, y_reserva)

# Estimar el área bajo la curva ROC (AUC)
logreg_reserva_auc_score = roc_auc_score(y_reserva, logreg_reserva.predict_proba(X_reserva_normed)[:, 1])
print(f"AUC score: {logreg_reserva_auc_score:.4f}")
#%%
from sklearn.model_selection import GridSearchCV

# Configurar la búsqueda de cuadrícula
param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'fit_intercept': [True, False],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter': [100, 1000, 2500, 5000]
}

# Crear la instancia de GridSearchCV
logreg_cv = GridSearchCV(LogisticRegression(), param_grid, scoring='roc_auc', cv=5, verbose=True, n_jobs=-1)

# Ajustar el modelo
logreg_cv.fit(X_test_normed, y_test)

# Imprimir el mejor parámetro y AUC calculado
print(f"Mejores parámetros: {logreg_cv.best_params_}")
print(f"AUC score: {logreg_cv.best_score_:.4f}")
#%%
# Copiar los datos originales y de reserva
X_original_normed = X_original.copy()
X_reserva_normed = X_reserva.copy()

# Log-normalizar la columna con mayor varianza
for df in [X_original_normed, X_reserva_normed]:
    df['monetary_log'] = np.log(df[col_to_normalize])
    df.drop(col_to_normalize, axis=1, inplace=True)

# Verificar la varianza después de la normalización
print(X_reserva_normed.var().round(3))
