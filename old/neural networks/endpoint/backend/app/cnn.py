#cnn.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Definir la arquitectura CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Primera capa convolucional: entrada 1x28x28 -> salida 32x28x28
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # Primera capa de pooling: 32x28x28 -> 32x14x14
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Segunda capa convolucional: 32x14x14 -> 64x14x14
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Segunda capa de pooling: 64x14x14 -> 64x7x7
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Capa completamente conectada: 64*7*7 -> 128
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # Capa de salida: 128 -> 10 (clases de MNIST)
        self.fc2 = nn.Linear(128, 10)
        # Funciones de activación y regularización
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        # Bloque convolucional 1
        x = self.conv1(x)  # Convolución
        x = self.relu(x)   # Activación
        x = self.pool1(x)  # Pooling
        
        # Bloque convolucional 2
        x = self.conv2(x)  # Convolución
        x = self.relu(x)   # Activación
        x = self.pool2(x)  # Pooling
        
        # Aplanar para las capas completamente conectadas
        x = x.view(-1, 64 * 7 * 7)
        
        # Capas completamente conectadas
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Función para entrenar el modelo
def train_model(model, train_loader, test_loader, epochs=5, learning_rate=0.001):
    # Definir función de pérdida y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Listas para guardar estadísticas de entrenamiento
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    # Bucle de entrenamiento
    for epoch in range(epochs):
        model.train()  # Modo entrenamiento
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Entrenamiento por lotes
        for i, (inputs, labels) in enumerate(train_loader):
            # Poner gradientes a cero
            optimizer.zero_grad()
            
            # Propagación hacia adelante
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Retropropagación y optimización
            loss.backward()
            optimizer.step()
            
            # Estadísticas
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Imprimir estadísticas cada 100 lotes
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        # Calcular pérdida y precisión en el conjunto de entrenamiento
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Evaluación en el conjunto de prueba
        model.eval()  # Modo evaluación
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        # Calcular pérdida y precisión en el conjunto de prueba
        test_loss = test_loss / len(test_loader)
        test_accuracy = 100 * test_correct / test_total
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')
    
    return train_losses, train_accuracies, test_losses, test_accuracies

# Función para visualizar los resultados
def plot_results(train_losses, train_accuracies, test_losses, test_accuracies):
    plt.figure(figsize=(12, 5))
    
    # Gráfico de pérdida
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss por Época')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    
    # Gráfico de precisión
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Precisión por Época')
    plt.xlabel('Época')
    plt.ylabel('Precisión (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

# Función principal
def main():
    # Configuración de dispositivo (GPU si está disponible)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Usando dispositivo: {device}')
    
    # Transformaciones para los conjuntos de datos
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Media y desviación estándar de MNIST
    ])
    
    # Cargar conjuntos de datos MNIST
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    
    # Crear data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Instanciar el modelo CNN y moverlo al dispositivo
    model = CNN().to(device)
    print(model)
    
    # Mover los datos de entrada y etiquetas al dispositivo durante el entrenamiento
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        print(f'Forma de los datos de entrada: {X.shape}')
        print(f'Forma de las etiquetas: {y.shape}')
        break
    
    # Entrenar el modelo
    train_losses, train_accuracies, test_losses, test_accuracies = train_model(
        model, train_loader, test_loader, epochs=5, learning_rate=0.001
    )
    
    # Visualizar los resultados
    plot_results(train_losses, train_accuracies, test_losses, test_accuracies)
    
    # Guardar el modelo entrenado
    torch.save(model.state_dict(), 'mnist_cnn_model.pth')
    print('Modelo guardado como mnist_cnn_model.pth')

if __name__ == '__main__':
    main()