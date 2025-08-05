import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

def show_image(image_id):
    img_path = f'images/{image_id}.jpg'
    img = Image.open(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def convert_to_gray(image_id):
    img_path = f'images/{image_id}.jpg'
    img = Image.open(img_path)
    img_rgb = np.array(img)
    img_gray = rgb2gray(img_rgb)
    return img_gray

def extract_hog_features(image_id):
    img_gray = convert_to_gray(image_id)
    features, hog_image = hog(img_gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

def extract_and_flatten_features(image_id):
    features = extract_hog_features(image_id)
    return features.flatten()

data = pd.read_csv('labels.csv')

X = []
y = []

for image_id, label in zip(data['id'], data['genus']):
    features = extract_and_flatten_features(image_id)
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

model = SVC(kernel='linear', probability=True)
model.fit(X_train_pca, y_train)

y_pred = model.predict(X_test_pca)

report = classification_report(y_test, y_pred)

print(report)
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test_pca)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('Tasa de falsos positivos')
plt.ylabel('Tasa de verdaderos positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()

fig, axes = plt.subplots(1, 5, figsize=(15, 5))

def label(value):
    if value == 1:
        return 'Abeja'
    else:
        return 'Abejorro'

for i, ax in enumerate(axes):
    if i < len(y_pred): 
        image_id = data['id'].iloc[i]
        img_path = f'images/{image_id}.jpg'
        img = Image.open(img_path)
        img_gray = rgb2gray(np.array(img))
        
        ax.imshow(img_gray, cmap='gray')
        ax.set_title(f"True: {label(y_test[i])} Pred: {label(y_pred[i])}")
        ax.axis('off') 

plt.tight_layout()
plt.savefig('output/predictions_comparison.png')
