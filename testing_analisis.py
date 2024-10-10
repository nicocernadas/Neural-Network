import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# https://youtu.be/iX_on3VxZzk?si=ONpvhSIVrYU9l_SA
#https://pandas.pydata.org/docs/reference/frame.html
#https://numpy.org/doc/stable/user/index.html << https://joserzapata.github.io/courses/python-ciencia-datos/numpy/

# pd.set_option('display.max_rows', None)

# DataFrame sin valores Null
df = pd.read_csv('water_potability.csv', sep=',')

# Reemplaza los nan por la mediana
df.fillna(df.median(), inplace=True)


# ============================================ Limpieza de Atipicos y Normalizacion ================================================== #
# Creo que esto no es una buena idea.
# No esta mal llenar los NaN con la mediana, pero si sacamos los valores atipicos perdemos mucha data.

def atipicos_col(col_name):
    sample_ordenado = sorted(col_name)
    n = len(sample_ordenado)

    # Cuartiles
    Q1 = sample_ordenado[n // 4]
    Q2 = (sample_ordenado[n // 2 - 1] + sample_ordenado[n // 2]) / 2
    Q3 = sample_ordenado[3 * n // 4]
    iqr = Q3 - Q1

    # print(f'Cuantiles de {col_name.values}')
    # print('Valores mayores a: ', Q3 + (1.5 * iqr), ' => Son atipicos')
    # print('Valores menores a: ',Q1 - (1.5 * iqr), ' => Son atipicos')
    # print('\n')

    atipicos = []

    # Calcula los valores atipicos
    for x in sample_ordenado:
        if (x > Q3 + (1.5 * iqr) or (x < Q1 - (1.5 * iqr))):
            atipicos.append(x)
        else:
            pass
    
    # Retorna la lista ordenada para despues armar el nuevo dataframe.
    return atipicos

def limpieza_col(data_frame):
    columnas = data_frame.columns.to_list()
    del columnas[0], columnas[6]
    atipicos = 0

    for item in columnas:
        atipicos = atipicos_col(data_frame[item])
        for x in atipicos:
            data_frame.loc[data_frame[item] == x, item] = data_frame[item].median() # Los atipicos se llenan con la mediana

    return data_frame

# Llamado a la funcion
df = limpieza_col(df)

# ============================================ Estandarizacion ================================================== #

def estandarizacion(data_frame):
    columnas = data_frame.columns.to_list()

    for item in columnas:
        # media
        media = data_frame[item].mean()
        # desviacion estandar
        desv_std = data_frame[item].std()
        # lista para guardar los valores escalados
        valores_esc = []
        for value in data_frame[item]:
            # estandariza
            valor_esc = (value - media) / desv_std
            # guarda
            valores_esc.append(valor_esc)
        # mete toda la lista en la columna
        data_frame[item] = valores_esc
    
    return data_frame

df = estandarizacion(df)

# =============================================================================================================== #

# ====================================== ERROR^2 =========================================== #

# Prueba con una línea dada
# m = 1.93939
# b = 4.73333
# sum_of_squares = 0.0
# df_copy = df[['ph', 'Potability']].copy()
# # calcular la suma de cuadrados
# for p in df_copy.itertuples(index=False):
#     y_salida = p[1]
#     y_predict = m*p[0] + b
#     residual_squared = (y_predict - y_salida)**2
#     sum_of_squares += residual_squared
# print(f"suma de cuadrados = {sum_of_squares}")

# ========================= GRAFICOS POST LIMPIEZA ============================ #
# Son horribles

# def regresion_lin(df, column):
#     plt.figure(figsize=(8, 5))
    
#     # Todas excepto potabilidad
#     data = df[[column, 'Potability']].dropna()
#     X = data['Potability'].values.reshape(-1, 1)
#     y = data[column].values
    
#     # Arma la regresion
#     model = LinearRegression()
#     model.fit(X, y)
#     y_pred = model.predict(X)
    
#     # Puntos
#     plt.scatter(X, y, color='blue', label='Data')
    
#     # Regresion lineal
#     plt.plot(X, y_pred, color='red', label='Linear Regression')
    
#     # Titulos y Ejes
#     plt.title(f'{column} and Potability')
#     plt.xlabel('Potability')
#     plt.ylabel(column)
    
#     plt.show()

# # Esto solo itera para printear todas juntas
# ploteo_total = df.columns[:-1]
# for column in ploteo_total:
#     regresion_lin(df, column)

# plt.figure(figsize=(10, 8))
# matriz_corr = df.corr()

# sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', linewidths=0.3)
# plt.title('Matriz de Correlacion')
# plt.show()

# ============================================== RED NEURONAL ========================================= #

from sklearn.model_selection import train_test_split

# Extraigo las columnas de entrada
inputs = df.iloc[:, 0:9].values
outputs = df.iloc[:, -1].values

# Conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size=1/3)

#shape retorna una tupla con las dimensiones de la matriz = (filas, columnas).
# por lo que shape[0], nos retorna las filas de la matriz.
n = x_train.shape[0] # número de registros de entrenamiento

# Red neuronal
# Capa oculta 1 (9 entradas, 100 neuronas)
w_hidden_1 = np.random.rand(100, 9)
# Sesgos
b_hidden_1 = np.random.rand(100, 1)

# Capa oculta 2 (100 entradas, 50 neuronas)
w_hidden_2 = np.random.rand(50, 100)
# Sesgos
b_hidden_2 = np.random.rand(50, 1)

# Capa de salida (50 entradas, 1 neurona)
w_output = np.random.rand(1, 50)
# Sesgo
b_output = np.random.rand(1, 1)

# Funciones de Activacion
relu = lambda x: np.maximum(x, 0)
sigmoide = lambda x: 1 / (1 + np.exp(-x))

# PREGUNTAR
# Como aplicar las 2 capas de neuronas a el forward prop
# Seria algo asi?

def f_prop(X):
    # Capa oculta 1
    z1 = w_hidden_1 @ X + b_hidden_1
    a1 = relu(z1)
    
    # Capa oculta 2
    z2 = w_hidden_2 @ a1 + b_hidden_2
    a2 = relu(z2)
    
    # Capa de salida
    z3 = w_output @ a2 + b_output 
    a3 = sigmoide(z3)

    return z1, a1, z2, a2, z3, a3

test_predictions = f_prop(x_test.transpose())[4] # me interesa solo la capa de salida, A2
test_comparisons = np.equal((test_predictions >= .5).flatten().astype(int), y_test)
accuracy = sum(test_comparisons.astype(int) / x_test.shape[0])

# por algun motivo me da cero....
print("ACCURACY: ", accuracy)