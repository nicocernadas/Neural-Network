import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from math import sqrt

# https://youtu.be/iX_on3VxZzk?si=ONpvhSIVrYU9l_SA
#https://pandas.pydata.org/docs/reference/frame.html
#https://numpy.org/doc/stable/user/index.html << https://joserzapata.github.io/courses/python-ciencia-datos/numpy/

# pd.set_option('display.max_rows', None)

# DataFrame sin valores Null
df = pd.read_csv('./water_potability.csv', sep=',')

# Reemplaza los nan por la mediana
df.fillna(df.median(), inplace=True)

# ============================================ Estandarizacion ================================================== #

def estandarizacion(data_frame):
    columnas = data_frame.columns.to_list()

    for item in columnas[:-1]:
        # media
        media = data_frame[item].mean()
        # desviacion estandar
        desv_std = data_frame[item].std()
        # lista para guardar los valores escalados
        valores_esc = []
        for value in data_frame[item]:
            # estandariza
            valor_esc = (value - media) / desv_std
            # Quiero valores negativos? No, no?
            valor_esc = valor_esc**2
            # valor_esc = sqrt(valor_esc)
            # guarda
            valores_esc.append(valor_esc)
        # mete toda la lista en la columna
        data_frame[item] = valores_esc
    
    return data_frame

df = estandarizacion(df)

# =============================================================================================================== #

# ============================================ Estandarizacion ================================================== #

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
    atipicos = 0

    for item in columnas[:-1]:
        atipicos = atipicos_col(data_frame[item])
        for x in atipicos:
            data_frame.loc[data_frame[item] == x, item] = data_frame[item].median() # Los atipicos se llenan con la mediana

    return data_frame

# Llamado a la funcion
df = limpieza_col(df)

# =============================================================================================================== #