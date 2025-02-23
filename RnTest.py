# Ejemplo de red neuronal simple con TensorFlow y Keras

# Importamos las librerías necesarias
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 1. Cargar el dataset MNIST
# MNIST contiene 70,000 imágenes de dígitos (28x28 píxeles): 60,000 para entrenamiento y 10,000 para prueba.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Preprocesar los datos
# Normalizamos los valores de píxel, que van de 0 a 255, a un rango de 0 a 1.
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convertimos las etiquetas (números del 0 al 9) en vectores one-hot.
# Por ejemplo, la etiqueta 3 se convierte en [0, 0, 0, 1, 0, 0, 0, 0, 0, 0].
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 3. Definir la arquitectura del modelo
# Utilizamos un modelo secuencial que consiste en:
# - Una capa Flatten para convertir la imagen 2D en un vector 1D.
# - Una capa densa (fully connected) con 128 neuronas y función de activación ReLU.
# - Una capa de salida con 10 neuronas (una para cada dígito) y función softmax para obtener probabilidades.
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Transforma la imagen en un vector de 784 elementos
    Dense(128, activation='relu'),   # Capa oculta con activación ReLU
    Dense(10, activation='softmax')  # Capa de salida con activación softmax
])

# 4. Compilar el modelo
# Se define el optimizador, la función de pérdida y la métrica a evaluar (precisión).
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. Entrenar el modelo
# Se entrena el modelo usando los datos de entrenamiento durante 5 épocas y con un tamaño de batch de 32.
model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1)

# 6. Evaluar el modelo
# Se evalúa el rendimiento del modelo usando los datos de prueba.
loss, accuracy = model.evaluate(x_test, y_test)
print("Pérdida en el conjunto de prueba:", loss)
print("Precisión en el conjunto de prueba:", accuracy)
