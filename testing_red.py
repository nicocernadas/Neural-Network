# ====================================================================================================== #

'''
PREGUNTAS PARA LOS PROFES 😎🤓
    (NOTA)
    Vamos a ir poniendo flechitas en el codigo <- , si hacen ctrl + f y buscan eso van a encontrar donde estan especificamente las preguntas
    En los demas archivos tambien funciona :)


COSAS POR HACER
    1) Relleno en los NANS no potables con la media de los no potables. Idem para los potables (✅)
    2) revisar columnas con datos muy atipicos y nan (tirar filas o columans) (✅)
    3) En el archivo /.graficos.py, si lo corren se grafican todas las columnas de una, y la tabla de correlacion
        tambien, pero los graficos son bastante fuleros


'''

# ====================================================================================================== #

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import testing_analisis as ta

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)

df= pd.read_csv('water_potability.csv', sep=',')

# =================================================== MANEJO DEL DATAFRAME ========================================================== #

# Estos no son preguntas, pero les muestro como se va limpiando para la red
df_ceros = ta.descarte(df, 'Potability', 0) # <- arma un dataframe nuevo con las muestras no potables
df_unos = ta.descarte(df, 'Potability', 1) # <- arma un dataframe nuevo con las muestras potables
ta.carga_nans(df, df_ceros, df_unos) # <- carga los nans con las medianas dependiendo si es o no potable
# Lei que es mejor limpiar primero y despues normalizar, pero estoy obteniendo mejores resultados si limpio despues
df = ta.normalizacion(df) # <- normaliza los datos
df = ta.limpieza_col(df) # <- borra los atipicos (directamente elimina la fila)

# ================================================================================================================================== #

# ===================================================== SEPARACION TRAIN/TEST ========================================================== #

# Extraigo las columnas de entrada
inputs = df.iloc[:, 0:9].values
outputs = df.iloc[:, -1].values

# Conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size=1/3)

#shape retorna una tupla con las dimensiones de la matriz = (filas, columnas).
# por lo que shape[0], nos retorna las filas de la matriz.
n = x_train.shape[0] # número de registros de entrenamiento

# ===================================================================================================================================== #

# ==================================================== RED NEURONAL ============================================================== #

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

# ===================================================================================================================================== #

# =========================================================== EJECUCION ==================================================================== #

L = .001
# Ejecutar descenso de gradiente
for i in range(500_000):
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

# ===================================================================================================================================== #

# ================================================= TESTEOS DE ACCURACY ================================================================== #

# Accuracy
# Esto sigue siendo medio bajo, entre 40-45%, no termino de entender bien por que, 
# quiza un mejor ajuste en paso y epochs? ya probe varias veces con valores distintos e igual no cambia demasiado <-
test_predictions = f_prop(x_test.transpose())[3]
test_comparisons = np.equal((test_predictions >= .5).flatten().astype(int), np.array(y_test >= .5).astype(int))
accuracy = sum(test_comparisons.astype(int) / x_test.shape[0])
print("TEST ACCURACY: ", accuracy)

# Arreglado ;), ahora da mejor el accuracy, ya no es del 200% jaja, pero tampoco es del 100%, queda en 80-82% <-
test_predictions = f_prop(x_train.transpose())[3]
test_comparisons = np.equal((test_predictions >= .5).flatten().astype(int), np.array(y_train >= .5).astype(int))
accuracy = sum(test_comparisons.astype(int) / x_test.shape[0])
print("TRAIN ACCURACY: ", accuracy)

# ===================================================================================================================================== #