from PIL import Image
import torch
from torch import nn
from torchvision import transforms
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware  # Importar CORS

# Definir la arquitectura correcta del modelo
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dp = nn.Dropout(p=0.4)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dp(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dp(x)
        x = self.fc3(x)
        return x

# Inicializar el modelo con la arquitectura correcta
input_size = 28*28
hidden_size = 128
output_size = 10

# Crear la instancia del modelo
model = MLP(input_size, hidden_size, output_size)

# Cargar los parámetros guardados
state_dict = torch.load("mnist_model.pth", map_location=torch.device("cpu"))
model.load_state_dict(state_dict)

# Poner el modelo en modo evaluación
model.eval()

# Transformaciones para preprocesar las imágenes
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Configurar FastAPI
app = FastAPI()

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
    return {"message": "Hello World"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
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
            _, predicted = torch.max(output, 1)
            prediction = predicted.item()
        
        print(f"Prediction: {prediction}")
        return {"prediction": prediction}
    except Exception as e:
        # Capturar y registrar cualquier error
        print(f"Error al procesar la imagen: {str(e)}")
        # Devolver error en formato JSON para que el cliente pueda manejarlo
        return {"error": str(e)}, 500

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=3000)