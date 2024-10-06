import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# https://youtu.be/iX_on3VxZzk?si=ONpvhSIVrYU9l_SA
#https://pandas.pydata.org/docs/reference/frame.html
#https://numpy.org/doc/stable/user/index.html << https://joserzapata.github.io/courses/python-ciencia-datos/numpy/

# pd.set_option('display.max_rows', None)

# DataFrame sin valores Null
df = pd.read_csv('water_potability.csv', sep=',')
ds = pd.read_csv('water_potability.csv', sep=',') # comparaciones

# Reemplaza los nan por la media
df.fillna(df.mean(), inplace=True)
ds.fillna(ds.mean(), inplace=True) # comparaciones

# ============================== Limpieza de Atipicos y Normalizacion ================================== #

def atipicos_col(col_name):
    sample_ordenado = sorted(col_name)
    n = len(sample_ordenado)

    # Cuartiles
    Q1 = sample_ordenado[n // 4]
    Q2 = (sample_ordenado[n // 2 - 1] + sample_ordenado[n // 2]) / 2
    Q3 = sample_ordenado[3 * n // 4]
    iqr = Q3 - Q1

    print('Cuantiles', )
    print('Valores mayores a: ', Q3 + (1.5 * iqr), ' => Son atipicos')
    print('Valores menores a: ',Q1 - (1.5 * iqr), ' => Son atipicos')
    print('\n')

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
    columnas = data_frame.columns
    atipicos = 0

    for item in columnas:
        atipicos = atipicos_col(data_frame[item])
        for x in atipicos:
            data_frame.loc[data_frame[item] == x, item] = np.nan

    return data_frame

df = limpieza_col(df)
print(df.describe())




