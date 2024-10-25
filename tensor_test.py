import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from keras.api import *
from keras.api.layers import Dense, BatchNormalization
from keras.api.optimizers import Adam

# Esto por algun motivo tira error, que onda? pero asi me deja ^^
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, BatchNormalization
# from tensorflow.keras.optimizers import Adam

# =========================== MODELO TENSORFLOW =============================== #
'''
MISMO ACCURACY QUE ANTES. YA ESTA NO SE DEBE PODER CONSEGUIR MAS CON ESTA CANTIDAD DE DATOS.



'''
# ============================================================================= #

# Cargar el archivo CSV
data = pd.read_csv('water_potability.csv')

# Separar características (X) y objetivo (y)
X = data.drop('Potability', axis=1).values  # 'Potability' es la columna objetivo
y = data['Potability'].values

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo
model = Sequential()

# Capa de normalización y capa oculta con ReLU
model.add(BatchNormalization(input_shape=(9,)))  # Normalización automática
model.add(Dense(6, activation='relu'))  # Capa oculta con 16 neuronas

# Capa de salida con función sigmoide para clasificación binaria
model.add(Dense(1, activation='sigmoid'))

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Pérdida: {loss}, Precisión: {accuracy}')
