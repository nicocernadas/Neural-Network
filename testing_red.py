# ====================================================================================================== #

'''
PREGUNTAS
    1) Asi como esta, el accuracy aca es el mismo que en la red de Scikit. Esto quiere decir que no se puede llegar a mucho mas?
        No se cuantas cosas mas haria con los datos para no modificar las muestras.

COSAS POR HACER
    1) Relleno en los NANS no potables con la media de los no potables. Idem para los potables
    2) revisar columnas con datos muy atipicos y nan (tirar filas o columans)

'''

# ====================================================================================================== #

# ====================================== RED NEURONAL ================================================== #

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import testing_analisis as ta
# from testing_analisis import df as df

df= pd.read_csv('water_potability.csv', sep=',')
print(df.describe())
# df.fillna(df.median(), inplace=True)

# Aca da rarisimo, a veces 0.129, 0.0, 0.9999
# Sin el * 2 - 1, con esto se va a 0.0 o 0.11
# df = ta.normalizacion(df)

# Este directamente da todo bajo, 0.157, 0.108, 0.08
# df = ta.estandarizacion(df)

# Extraigo las columnas de entrada
inputs = df.iloc[:, 0:9].values
outputs = df.iloc[:, -1].values

# Conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size=1/3)
print(y_train)

#shape retorna una tupla con las dimensiones de la matriz = (filas, columnas).
# por lo que shape[0], nos retorna las filas de la matriz.
n = x_train.shape[0] # número de registros de entrenamiento

# Red neuronal
np.random.seed(0)
# pesos
w_hidden_1 = np.random.rand(9,9)
w_output_1 = np.random.rand(1,9)

# sesgos
b_hidden = np.random.rand(9,1)
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

# Derivadas de las funciones de activación
d_relu = lambda x: x > 0
d_sigmoide = lambda x: np.exp(-x) / (1 + np.exp(-x)) ** 2

# Devuelve pendientes para pesos y sesgos
# usando la regla de la cadena
def b_prop(z1, a1, z2, a2, X, Y):
    dC_dA2 = 2 * a2 - 2 * Y
    dA2_dZ2 = d_sigmoide(z2)
    dZ2_dA1 = w_output_1
    dZ2_dW2 = a1
    dZ2_dB2 = 1
    dA1_dZ1 = d_relu(z1)
    dZ1_dW1 = X
    dZ1_dB1 = 1

    dC_dW2 = dC_dA2 @ dA2_dZ2 @ dZ2_dW2.T
    dC_dB2 = dC_dA2 @ dA2_dZ2 * dZ2_dB2
    dC_dA1 = dC_dA2 @ dA2_dZ2 @ dZ2_dA1
    dC_dW1 = dC_dA1 @ dA1_dZ1 @ dZ1_dW1.T
    dC_dB1 = dC_dA1 @ dA1_dZ1 * dZ1_dB1

    return dC_dW1, dC_dB1, dC_dW2, dC_dB2

L = .01
# Ejecutar descenso de gradiente
for i in range(200_000):
    # seleccionar aleatoriamente uno de los datos de entrenamiento
    idx = np.random.choice(n, 1, replace=False)
    X_sample = x_train[idx].transpose()
    Y_sample = y_train[idx]

    # pasar datos seleccionados aleatoriamente a través de la red neuronal
    Z1, A1, Z2, A2 = f_prop(X_sample)

    # distribuir error a través de la retropropagación
    # y devolver pendientes para pesos y sesgos
    dW1, dB1, dW2, dB2 = b_prop(Z1, A1, Z2, A2, X_sample, Y_sample)

    # actualizar pesos y sesgos
    w_hidden_1 -= L * dW1
    b_hidden -= L * dB1
    w_output_1 -= L * dW2
    b_output -= L * dB2

# Accuracy
test_predictions = f_prop(x_test.transpose())[3]
test_comparisons = np.equal((test_predictions >= .5).flatten().astype(int), np.array(y_test >= .5).astype(int))
accuracy = sum(test_comparisons.astype(int) / x_test.shape[0])
print("TEST ACCURACY: ", accuracy)

# test_predictions = f_prop(x_train.transpose())[3]
# test_comparisons = np.equal((test_predictions >= .5).flatten().astype(int), np.array(y_train >= .5).astype(int))
# accuracy = sum(test_comparisons.astype(int) / x_test.shape[0])
# print("TRAIN ACCURACY: ", accuracy)
