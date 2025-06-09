# Preprocesamiento
# Codificación de etiquetas
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import preprocess_input


# Definir ruta al CSV guardado en scripts
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'df_images.csv')
df = pd.read_csv(csv_path)

# Inicializamos labelencoder de sklearn, lo cual nos permitira codificar la ruta de las imagenes o imagenes en 0 y 1
label_encoder = LabelEncoder()
# Se crea una nueva columna 'category_encoded' con las etiquetas numéricas (ej. 0 para Healthy, 1 para Tumor)
df['category_encoded'] = label_encoder.fit_transform(df['label'])

print("DataFrame después de la codificación de etiquetas:")
print(df.head())
print(f"Clases codificadas: {label_encoder.classes_} -> {label_encoder.transform(label_encoder.classes_)}")

# Dividimos en entrenamiento (80%) y un conjunto temporal (20% para validación + prueba) para evitar el data leakage
X_train_original, X_temp, y_train_original, y_temp = train_test_split(
    df[['image_path']],  # Características (rutas de imagen como DataFrame)
    df['category_encoded'], # Etiquetas numéricas
    train_size=0.8,         # 80% para entrenamiento
    shuffle= False,         # Mezclar los datos antes de dividir
    random_state=42,        # Para reproducibilidad
    stratify=df['category_encoded'] # Asegurar proporciones de clase similares en la división
)

# Luego, dividir el conjunto temporal en validación (50% de temp -> 10% del total)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp,                 # DataFrame con rutas de imagen del conjunto temporal
    y_temp,                 # Etiquetas del conjunto temporal
    test_size=0.5,          # 50% de X_temp para el conjunto de prueba (el resto para validación)
    shuffle = False,
    random_state=42,
    stratify=y_temp         # Estratificar sobre las etiquetas del conjunto temporal
)

# Balancemos ahora si los datos de entrenamiento usando el método de RandomOversampler
ros = RandomOverSampler(random_state=42)
# Desempaquetamos variables y realizamos el balanceo
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_original, y_train_original)

# train_df utilizará los datos de entrenamiento sobremuestreados.
train_df = pd.DataFrame(X_train_resampled, columns=['image_path'])
train_df['category_encoded'] = y_train_resampled.astype(str)

# valid_df y test_df utilizan los datos originales de validación y prueba, sin sobremuestreo.
valid_df = pd.DataFrame(X_valid, columns=['image_path'])
valid_df['category_encoded'] = y_valid.astype(str)

test_df = pd.DataFrame(X_test, columns=['image_path'])
test_df['category_encoded'] = y_test.astype(str)

#  Generadores de datos para el ResNet
batch_size = 8 # Lotes de 8
img_size = (224, 224)  # ResNet fue entrenado con 224*224, por ende debemos pasarle la información con está estructura

# Data augmentation para entrenamiento
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Preprocesamiento específico de ResNet
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Solo preprocesamiento para validación y prueba
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Generador para el conjunto de entrenamiento
train_gen = train_datagen.flow_from_dataframe(
    train_df,                   # DataFrame con datos de entrenamiento (sobremuestreados)
    x_col='image_path',         # Columna con las rutas de las imágenes
    y_col='category_encoded',   # Columna con las etiquetas codificadas (como string)
    target_size=img_size,       # Tamaño al que se redimensionarán las imágenes
    class_mode='binary',        # Para clasificación binaria
    color_mode='rgb',           # Cargar imágenes en color
    shuffle = False,            # Mezclar los datos de entrenamiento en cada época
    batch_size=batch_size
)

# Generador para el conjunto de validación
valid_gen = test_datagen.flow_from_dataframe(
    valid_df,                   # DataFrame con datos de validación (originales)
    x_col='image_path',
    y_col='category_encoded',
    target_size=img_size,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False,               # Mezclar datos de validación
    batch_size=batch_size
)

# Generador para el conjunto de prueba
test_gen = test_datagen.flow_from_dataframe(
    test_df,                    # DataFrame con datos de prueba (originales)
    x_col='image_path',
    y_col='category_encoded',
    target_size=img_size,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False,              # IMPORTANTE: No mezclar el conjunto de prueba
    batch_size=batch_size
)

# Obtener un batch del generador de entrenamiento
images, labels = next(train_gen)

plt.figure(figsize=(12, 6))
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(images[i].astype("uint8"))
    plt.title(f"Label: {labels[i]}")
    plt.axis("off")

# Guardar antes de mostrar (o sin mostrar si estás en WSL)
plt.savefig("sample_batch.png")  # Guardar el gráfico
print("Imagen guardada como sample_batch.png")


def obtener_generadores():
    return train_gen, valid_gen, test_gen, label_encoder
