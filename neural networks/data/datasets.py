from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),

])

def get_mnist():
    train_dataset = datasets.MNIST(root='./data', train = True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train = False, download=True, transform=transform)
    return train_dataset, test_dataset
def get_iris():
    data = load_iris()
    X = data.data
    y = data.target.reshape(-1,1)
    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # X_train, X_test, y_train , y_test = 
    return train_test_split(X, y, test_size=0.2, shuffle=True)
