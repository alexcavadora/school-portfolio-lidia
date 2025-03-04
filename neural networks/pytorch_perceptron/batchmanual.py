from data.datasets import get_iris
from pytorch_perceptron.pytorch_utils import get_device
from pytorch_perceptron.models import MLP
import torch
import torch.nn as nn
import torch.optim as optim
import time

X_train, X_test, y_train , y_test = get_iris()
device = torch.device("cpu")#get_device()

X_train = torch.tensor(X_train).to(device, dtype=torch.float32)
X_test = torch.tensor(X_test).to(device, dtype=torch.float32)
y_train = torch.tensor(y_train).to(device, dtype=torch.float32)
y_test = torch.tensor(y_test).to(device, dtype=torch.float32)

input_size = 4
hidden_size = 10
output_size = 3
epochs = 10000
learning_rate = 0.1

batch_size = 16 # to add noise in the dataset and avoid overfitting like k-folds

model = MLP(input_size, hidden_size, output_size)
model.to(device)
print(model)

loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

history = {'train_loss': [],
            'accuracy': [],
            'val_loss': [],
            'epoch': []}

st = time.time()
#training cl
num_samples = X_train.shape[0]
for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(num_samples)
    epoch_loss = 0
    for i in range(0, num_samples, batch_size):
        index = permutation[i: i+batch_size]
        batch_x, batch_y = X_train[index], y_train[index]
        optimizer.zero_grad() #reiniciar gradientes
        outputs = model(batch_x) # forward
        loss = loss_function(outputs, batch_y) #calcular la pérdida
        loss.backward() # backward -> calcular gradientes
        optimizer.step()
        epoch_loss += loss.item()
    train_loss = epoch_loss/(num_samples//batch_size)
    history['train_loss'].append(train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        val_outputs = model(X_test)
        val_loss = loss_function(val_outputs, y_test).item()
        history['val_loss'].append(val_loss)

    if epoch%20== 0:
        print(f"epoch = {epoch}, val_loss = {val_loss}, train_loss = {train_loss}")

ft = time.time()
#evluation
model.eval()
with torch.no_grad(): #desactivar el cálculo de los gradientes
    predictions = model(X_test)
    predicted_labels = torch.argmax(predictions, dim=1)
    y_test_labels = torch.argmax(y_test, dim=1)
    accuracy = torch.mean((predicted_labels == y_test_labels).float())
    print(f"Accuracy: {accuracy}")
print(f"Time taken = {ft-st:0.2f} seconds")

import matplotlib.pyplot as plt

plt.figure(0)
plt.plot(history['train_loss'], label="Train Loss")
plt.plot(history['val_loss'], label="Validation Loss")
plt.show()