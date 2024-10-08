'''
PREGUNTAS

    1) Datos con NaN: que hacemos? Por que no se puede poner la media en cada uno, arruinaria las muestras de las distintas
    aguas, cambiando el valor final de salida. Los Valores con NaN se pueden procesar en una red igualmente? O les ponemos ceros

    2) 

COSAS POR HACER

    1) Terminar de limpiar el Data Frame.




'''

# ====================================== RED NEURONAL ================================================== #

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('water_potability.csv', sep=',')
df.fillna(df.median(), inplace=True)

# Extraigo las columnas de entrada
inputs = df.iloc[:, 0:9].values
outputs = df.iloc[:, -1].values

# Conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size=1/3)

#shape retorna una tupla con las dimensiones de la matriz = (filas, columnas).
# por lo que shape[0], nos retorna las filas de la matriz.
n = x_train.shape[0] # número de registros de entrenamiento

# Red neuronal
# pesos
w_hidden_1 = np.random.rand(3,9)
w_output_1 = np.random.rand(1,3)

# Como seria para ponerle 2 capas de neuronas?
# w_hidden_2 = np.random.rand(,)
# w_output_2 = np.random.rand(,)

# sesgos
b_hidden = np.random.rand(3,1)
b_output = np.random.rand(1,1)

# Funciones de Activacion
relu = lambda x: np.maximum(x, 0)
sigmoide = lambda x: 1 / (1 + np.exp(-x))

def f_prop(X):
    z1 = w_hidden_1 @ X + b_hidden
    a1 = relu(z1)
    z2 = w_output_1 @ a1 + b_output
    a2 = sigmoide(z2)
    return z1, a1, z2, a2

test_predictions = f_prop(x_test.transpose())[3] # me interesa solo la capa de salida, A2
test_comparisons = np.equal((test_predictions >= .5).flatten().astype(int), y_test)
accuracy = sum(test_comparisons.astype(int) / x_test.shape[0])
print("ACCURACY: ", accuracy)
