from PIL import Image
import torch
from torch import nn
from torchvision import transforms
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Definir la arquitectura correcta del modelo
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

# Crear la instancia del modelo
model = CNN()

# Cargar los parámetros guardados
try:
    state_dict = torch.load("mnist_cnn_model.pth", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    print("Modelo cargado correctamente")
except Exception as e:
    print(f"Error al cargar el modelo: {str(e)}")

# Poner el modelo en modo evaluación
model.eval()

# Transformaciones para preprocesar las imágenes
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Media y desviación estándar de MNIST
])

# Configurar FastAPI
app = FastAPI(title="MNIST Digit Recognition API")

# Configurar CORS para permitir solicitudes desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.get("/")
async def root():
    """Endpoint raíz para verificar que el servidor está funcionando"""
    return {"message": "MNIST Digit Recognition API", "status": "online"}

@app.get("/health")
async def health_check():
    """Endpoint para verificar el estado del servidor"""
    return {"status": "healthy"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint para predecir el dígito en una imagen
    
    Args:
        file: Imagen que contiene un dígito escrito a mano
        
    Returns:
        Predicción del dígito (0-9)
    """
    try:
        # Leer la imagen
        image_bytes = await file.read()
        
        # Depuración: ver tamaño de los datos recibidos
        print(f"Recibidos {len(image_bytes)} bytes")
        
        # Abrir la imagen
        image = Image.open(io.BytesIO(image_bytes))
        
        # Aplicar transformaciones
        image_tensor = transform(image).unsqueeze(0)
        
        # Hacer predicción
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            _, predicted = torch.max(output, 1)
            prediction = predicted.item()
        
        # Convertir probabilidades a lista
        probs = probabilities.tolist()
        
        print(f"Prediction: {prediction}")
        return {
            "prediction": prediction,
            "confidence": float(probabilities[prediction]),
            "probabilities": {i: float(p) for i, p in enumerate(probs)}
        }
    
    except Exception as e:
        # Capturar y registrar cualquier error
        error_msg = f"Error al procesar la imagen: {str(e)}"
        print(error_msg)
        
        # Devolver error en formato adecuado
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=3000)