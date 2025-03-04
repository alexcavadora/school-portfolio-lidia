import numpy as np

class Perceptron:
    def __init__(self, input_size=2):  # Initialize weights and bias randomly
        self.__weights = np.random.randn(input_size)
        self.__bias = np.random.randn()

    def __activacion(self, x):  # Sigmoid activation function
        return 1 / (1 + np.exp(-x))

    def __dot(self, inputs):  # Dot product of inputs and weights, plus bias
        return np.dot(inputs, self.__weights) + self.__bias

    def forward(self, inputs):  # Forward pass: compute the output
        return self.__activacion(self.__dot(inputs))

    def fit(self, inputs, outputs, lr=0.1, epochs=1000):
        for epoch in range(epochs):
            for i in range(len(inputs)):
                input_data = inputs[i]
                target = outputs[i]

                # Forward pass
                output = self.forward(input_data)

                # Compute error (output - target)
                error = output - target

                self.__weights -= lr * error * input_data  
                self.__bias -= lr * error 

            # Print progress every 50 epochs
            if epoch % 50 == 0:
                print(f"Epoch: {epoch}, error: {error}")

# Test the perceptron
if __name__ == '__main__':
    p = Perceptron(2)
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    outputs = np.array([1, 0, 0, 0])  # Binary classification task
    p.fit(inputs, outputs, lr=0.5, epochs=1000)

    # Test the trained perceptron
    for input_data in inputs:
        output = p.forward(input_data)
        print(f"Input: {input_data}, Output: {output:2f}")