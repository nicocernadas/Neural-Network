import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from keras.api import *
from keras.api.layers import Dense, BatchNormalization
from keras.api.optimizers import Adam

# =========================== MODELO TENSORFLOW =============================== #

# Cargar el archivo CSV
data = pd.read_csv('./csvs/water_potability.csv')

# Separar caracteristicas (X) e (y)
X = data.drop('Potability', axis=1).values  # 'Potability' es la columna objetivo
y = data['Potability'].values

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo
model = Sequential()

# Capa de normalizacion y capa oculta con ReLU
model.add(BatchNormalization(input_shape=(9,))) # Normaliza solo
model.add(Dense(4, activation='relu'))  # Capa oculta con 4 neuronas

# Capa de salida con funcion sigmoide para clasificación
model.add(Dense(1, activation='sigmoid'))

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=.01), loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Evaluar el accuracy del modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Pérdida: {loss}, Precisión: {accuracy}')

# ==================================================================================== #