import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import math

# https://youtu.be/iX_on3VxZzk?si=ONpvhSIVrYU9l_SA
#https://pandas.pydata.org/docs/reference/frame.html
#https://numpy.org/doc/stable/user/index.html << https://joserzapata.github.io/courses/python-ciencia-datos/numpy/

df = pd.read_csv('water_potability.csv', sep=',')

# ============================================ Normalizacion ================================================== #

def normalizacion(data_frame):
    columnas = data_frame.columns.to_list()
    new_val = []
    for item in columnas[:-1]:
        max_value = data_frame[item].max()
        for value in data_frame[item]:   
            new_val.append(value/max_value)
        data_frame[item] = new_val
        new_val.clear()
    
    return data_frame

# =============================================================================================================== #

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
            valor_esc = ((value - media) / desv_std) 
            # guarda
            valores_esc.append(valor_esc)
        # mete toda la lista en la columna
        data_frame[item] = valores_esc
    
    return data_frame

# =============================================================================================================== #

# ============================================ Atipicos ================================================== #

def atipicos_col(values_list):
    sample_ordenado = sorted(values_list)
    n = len(sample_ordenado)

    Q1 = sample_ordenado[n // 4]
    Q2 = (sample_ordenado[n // 2 - 1] + sample_ordenado[n // 2]) / 2
    Q3 = sample_ordenado[3 * n // 4]
    iqr = Q3 - Q1

    # Esto ahora cambio un poco, ya no guardo los atipicos, se van a guardar las posiciones
    indices_atipicos = []

    # Esto retorna por cada lista, el indice (index) y el valor (x)
    for index, x in enumerate(sample_ordenado):
        if (x > Q3 + (1.5 * iqr) or (x < Q1 - (1.5 * iqr))):
            indices_atipicos.append(index)

    return indices_atipicos

def limpieza_col(data_frame):
    columnas = data_frame.columns.to_list()

    for item in columnas[:-1]:
        # Ahora se pasa cada columna como una lista
        indices_atipicos = atipicos_col(data_frame[item].to_list())
        
        # Si no esta vacia...
        if indices_atipicos:
            data_frame = data_frame.drop(data_frame.index[indices_atipicos])

    return data_frame

# =============================================================================================================== #

# ================================== DIVISION ENTRE POTABLES/NO POTABLES ====================================== #

# Esto esta pensado para sacar las medianas de las columnas cuando el agua es potable/no potable

# Recibe el dataframe, la columna y el valor que te quieras quedar
# Devuelve el nuevo Dataframe (copia modificada, a menos que se pase inplace=true)
def descarte(data_frame, columna, dato):
    indexes = data_frame[data_frame[columna] != dato].index
    return data_frame.drop(indexes)

# =============================== MEDIANAS DE UN DATAFRAME ========================================= #

def medians(data_frame):
    columnas = data_frame.columns.to_list()
    medians = []
    for item in columnas[:-1]:
        medians.append(data_frame[item].median())
    return medians

# ================================================================================================== #

# ========================================== CARGA DE MEDIANAS EN NANS ========================================== #

def carga_nans(data_frame, data_ceros, data_unos):
    columnas = data_frame.columns.to_list()
    # Esto devuelve:
    # index = indice de la fila
    # fila = todos los datos de la fila en un [[formato]]
    for index, fila in data_frame.iterrows():
        # Y por cada fila del dataframe, itero en las columnas
        for i in range(9):
            # Cuando i = 0 > fila['ph'], cuando i = 1 > fila['Hardness']
            if math.isnan(fila.iloc[i]) and fila.iloc[9] == 0:
                data_frame.loc[index, columnas[i]] = data_ceros[columnas[i]].median()
            elif math.isnan(fila.iloc[i]) and fila.iloc[9] == 1:
                data_frame.loc[index, columnas[i]] = data_unos[columnas[i]].median()

# ================================================================================================================= #
    


