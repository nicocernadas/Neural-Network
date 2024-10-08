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
# ds = pd.read_csv('water_potability.csv', sep=',') # comparaciones
# Reemplaza los nan por la mediana
df.fillna(df.median(), inplace=True)


# ============================================ Limpieza de Atipicos y Normalizacion ================================================== #

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

# Consultar por esto
# Columnas 'ph' y 'Trihalomethanes' vuelven todas en NaN, por eso las borro en la segunda linea de 'limpieza_col()'
df = limpieza_col(df)
# print('\nPOST LIMPIEZA\n',df.describe())

# ====================================== ERROR^2 =========================================== #

# Prueba con una línea dada
m = 1.93939
b = 4.73333
sum_of_squares = 0.0
df_copy = df[['ph', 'Potability']].copy()
# calcular la suma de cuadrados
for p in df_copy.itertuples(index=False):
    y_salida = p[1]
    y_predict = m*p[0] + b
    residual_squared = (y_predict - y_salida)**2
    sum_of_squares += residual_squared
print(f"suma de cuadrados = {sum_of_squares}")



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

# ======================================================================================= #



