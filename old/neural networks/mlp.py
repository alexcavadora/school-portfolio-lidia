# MLP = Multi Layer Perceptron
import numpy as np
class PerceptronLayer:
    def __init__(self, input_size, output_size):
        """Initialize a new PerceptronLayer with given dimensions
        
        Args:
            input_size (int): Number of input features
            output_size (int): Number of output neurons
        """
        self.W = np.random.randn(input_size, output_size)  # Weight matrix shape: (input_size, output_size)
        self.b = np.zeros((1, output_size))  # Bias vector shape: (1, output_size)

    @staticmethod
    def sigmoid(z):
        return 1/ (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(z):
        return z * (1-z)

    def forward(self, X):
        linear_output = np.dot(X, self.W) + self.b
        self.fc1 = self.sigmoid(linear_output)
        return self.fc1

    def backward(self, X, delta, learning_rate):
        d_output = delta * self.sigmoid_derivative(self.fc1)
        dW = np.dot(X.T, d_output)
        db = np.sum(d_output, axis = 0, keepdims=True)
        d_input = np.dot(d_output, self.W.T)
        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        return d_input



class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_layer = PerceptronLayer(input_size, hidden_size)
        self.output_layer = PerceptronLayer(hidden_size, output_size)
    
    def forward(self, X):
        self.fc1 = self.hidden_layer.forward(X)
        self.fc2 = self.output_layer.forward(self.fc1)
        return self.fc2
    
    def backward(self, X, y, lr):
        output_error = self.fc2 - y
        output_delta = output_error * PerceptronLayer.sigmoid_derivative(self.fc2)
        hidden_delta = self.output_layer.backward(self.fc1, output_delta, lr)
        self.hidden_layer.backward(X, hidden_delta, lr)
    
    def train(self, X, y, lr = 0.1, epochs = 10000):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, lr)
            if epoch % 1000 == 0:
                loss = np.mean((y - self.fc2)**2)
                print(f"Epoch: {epoch}, loss = {loss}")
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

data = load_iris()
X = data.data
y = data.target.reshape(-1,1)
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train , y_test= train_test_split(X, y, test_size=0.2, shuffle=True)

mlp = MLP(input_size=4, hidden_size= 10, output_size= 3)
mlp.train(X_train, y_train, lr= 0.1, epochs=10000)

predictions = mlp.predict(X_test)
y_test_labels = np.argmax(y_test, axis= 1)
accuracy = np.mean(predictions == y_test_labels)
print(accuracy)


