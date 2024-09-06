from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time

start_time = time.time()

data = pd.read_csv('output/database2.csv')
#print(data)

X = data.drop(['Target'], axis = 1)
y = data['Target']
# pca = PCA(n_components=10)
# X = pca.fit_transform(X)
# print(pca.explained_variance_ratio_)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(criterion='gini', splitter='best', random_state=69)
rf = RandomForestClassifier(n_estimators=50, random_state=42)

scores = cross_val_score(clf, X, y, cv=10)

print("Puntuaciones de cada fold: ", scores)
print("Puntuación media: ", scores.mean())

clf.fit(X_train, y_train)

test_score = clf.score(X_test, y_test)
print("Puntuación en el conjunto de prueba: ", test_score)

end_time = time.time()

elapsed_time = end_time - start_time
start_time = end_time

print(f"Tiempo de ejecución: {elapsed_time} segundos")


scores = cross_val_score(rf, X, y, cv=10)

print("Puntuaciones de cada fold: ", scores)
print("Puntuación media: ", scores.mean())

rf.fit(X_train, y_train)

test_score = rf.score(X_test, y_test)
print("Puntuación en el conjunto de prueba: ", test_score)


end_time = time.time()

elapsed_time = end_time - start_time

print(f"Tiempo de ejecución: {elapsed_time} segundos")
