from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# transform lo que hace es normalizar los datos de entrada, es decir, los valores de los pixeles de las imágenes
# para que estén en un rango de -1 a 1. Esto es importante para que el modelo pueda aprender de manera más eficiente.
# (0.5,) es la media y (0.5,) es la desviación estándar. Estos valores son los que se utilizan para normalizar los datos.
# Ejemplo: si un pixel tiene un valor de 100, después de la normalización tendrá un valor de -0.8.

def get_mnist():
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform) # Shape (60000, 28, 28)
    test_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform) # Shape (10000, 28, 28)
    return train_dataset, test_dataset

# Cargar datos
def get_iris():

    data = load_iris()
    X = data.data
    y = data.target.reshape(-1,1)

    # Preprocesamiento
    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, shuffle=True)

    return X_train, X_test, y_train, y_test	
