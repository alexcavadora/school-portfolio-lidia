import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data.datasets import get_mnist
from pytorch_perceptron.models import MLP
from pytorch_perceptron.pytorch_utils import get_device

device = get_device()
train_ds, test_ds = get_mnist()

input_size = 28 * 28
hidden_size = 64
output_size = 10
learning_rate = 0.1
batch_size = 64
epochs = 100

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

model = MLP(input_size, hidden_size, output_size).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

history = {'train_loss': [], 'test_loss': []}
start_time = time.time()

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = loss_function(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    history['train_loss'].append(epoch_loss / len(train_loader))

    model.eval()
    with torch.no_grad():
        epoch_loss = 0
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            val_outputs = model(X_batch)
            val_loss = loss_function(val_outputs, y_batch).item()
            epoch_loss += val_loss
        history['test_loss'].append(epoch_loss / len(test_loader))

    if (epoch % 10) == 0:
        print(f"Epoch {epoch}: Train loss {history['train_loss'][-1]}, Test loss {history['test_loss'][-1]}")


end_time = time.time()
torch.save(model.state_dict(), "mnist_mlp.pth")
print("Model saved as mnist_mlp.pth")

plt.figure()
plt.plot(history['train_loss'], label='Train loss')
plt.plot(history['test_loss'], label='Test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
