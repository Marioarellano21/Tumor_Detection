import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# Librerías de Scikit-learn para preprocesamiento y métricas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Imbalanced-learn para manejar desbalance de clases
from imblearn.over_sampling import RandomOverSampler

# Componentes de TensorFlow y SKeras
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50 # El modelo base que usaremos
from tensorflow.keras.applications.resnet import preprocess_input # Función de preprocesamiento específica para ResNet
from tensorflow.keras.metrics import AUC # Métrica de Área Bajo la Curva ROC



# Configuración para la visualización y logs de TensorFlow
sns.set_style("whitegrid") # Estilo para los gráficos de Seaborn
tf.get_logger().setLevel('ERROR') # Suprime mensajes informativos de TensorFlow, mostrando solo errores

# Carga de datos
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_path = os.path.join(project_root, "images") # Contruimos ruta absoluta de la carpeta image donde están las imagenes
categories = ["Healthy", "Tumor"] # Aquí están el nombre de las subcarpetas las cuales representa las distintas clases

image_paths = []  # Lista para almacenar las rutas a cada imagen
labels = []       # Lista para almacenar la etiqueta de cada imagen

print("Ruta absoluta esperada para la carpeta 'images':", base_path)
print("¿Existe la carpeta 'images'? ->", os.path.isdir(base_path))
print("Contenido de 'images':", os.listdir(base_path) if os.path.isdir(base_path) else "No existe")


# Recorremos cada carpeta y catergoría
for category in categories:
    category_path = os.path.join(base_path, category) # Construir la ruta a la carpeta de la categoría
    if os.path.isdir(category_path):
        # Iterar sobre cada archivo de imagen dentro de la carpeta de la categoría
        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name) # Ruta completa a la imagen
            image_paths.append(image_path) # Añadiendo la ruta de la imagen a la lista
            labels.append(category)        # Añadiendo nombre de la categoría a la lista
    else:
        print(f"Advertencia: El directorio para la categoría '{category}' no fue encontrado en '{category_path}'")

# Crear un DataFrame de Pandas para almacenar las rutas de las imágenes y sus etiquetas
df = pd.DataFrame({"image_path": image_paths, "label": labels})
# Guardar df en CSV dentro de la carpeta scripts
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'df_images.csv')
df.to_csv(output_path, index=False)

# Mostrar las primeras filas del DataFrame y la distribución de clases
print("DataFrame inicial con rutas de imágenes y etiquetas:")
print(df.head())
print("\nDistribución de clases inicial:")
print(df['label'].value_counts())

