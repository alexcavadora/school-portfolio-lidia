from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import pandas as pd
import time

start_time = time.time()

# Cargar los datos
data = pd.read_csv('output/database2.csv')

# Separar características y target
X = data.drop(['Target'], axis=1)
y = data['Target']

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definir los clasificadores
clf = DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=2, min_samples_split=8, splitter='best', random_state=69)
rf = RandomForestClassifier(n_estimators=40, random_state=42)
ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=4),algorithm='SAMME', n_estimators=50, random_state=42)

# Evaluar DecisionTreeClassifier con validación cruzada
scores = cross_val_score(clf, X, y, cv=10)
print("Puntuación media de DecisionTree: ", scores.mean())

# clf.fit(X_train, y_train)
# test_score = clf.score(X_test, y_test)
# print("Puntuación en el conjunto de prueba de DecisionTree: ", test_score)

# Evaluar RandomForestClassifier con validación cruzada
scores = cross_val_score(rf, X, y, cv=10)
print("Puntuación media de RandomForest: ", scores.mean())

# rf.fit(X_train, y_train)
# test_score = rf.score(X_test, y_test)
# print("Puntuación en el conjunto de prueba de RandomForest: ", test_score)

# Evaluar AdaBoostClassifier con validación cruzada
scores = cross_val_score(ada, X, y, cv=10)
print("Puntuación media de AdaBoost: ", scores.mean())

# ada.fit(X_train, y_train)
# test_score = ada.score(X_test, y_test)
# print("Puntuación en el conjunto de prueba de AdaBoost: ", test_score)

# Medir el tiempo de ejecución
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tiempo de ejecución: {elapsed_time} segundos")
