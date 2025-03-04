import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Cargar el modelo preentrenado sin la parte del clasificador superior
base_model = VGG16(weights='imagenet', include_top=False)

#% Definir la capa de la que se extraerán las características
layer_name = 'block1_conv1' #% Puedes cambiar a otra capa para características de bajo o alto nivel
feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)

# Ruta al directorio donde se encuentran las imágenes organizadas en carpetas por categoría
image_dir = 'munster' # Reemplaza con tu ruta

# Configurar el generador de imágenes
datagen = ImageDataGenerator(rescale=1.0/255.0)

# Cargar las imágenes desde el directorio
batch_size = 32
image_size = (224, 224) # Tamaño requerido por VGG16
generator = datagen.flow_from_directory(
    image_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Extraer características de todas las imágenes utilizando el modelo
features = feature_extractor.predict(generator, verbose=1)

# Obtener las etiquetas correspondientes a las imágenes
labels = generator.classes

# Mostrar la dimensión de las características extraídas
print('Dimensión de las características extraídas:', features.shape)

# Guardar las características y etiquetas para uso posterior
np.save('caracteristicas_extraidas.npy', features)
np.save('etiquetas.npy', labels)
