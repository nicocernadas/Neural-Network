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
    new_df = data_frame.copy() # <-
    columnas = new_df.columns.to_list()
    new_val = []
    for item in columnas[:-1]:
        max_value = new_df[item].max()
        for value in new_df[item]:   
            new_val.append(value/max_value)
        new_df[item] = new_val
        new_val.clear()
    
    return new_df

# =============================================================================================================== #

# ============================================ Estandarizacion ================================================== #

# NUEVO! <- Estandarizacion
# Estandarizacion
def estandarizacion(data_frame):
    new_df = data_frame.copy()
    columnas = data_frame.columns.to_list()
    for item in columnas[:-1]:
        # media
        media = new_df[item].mean()
        # desviacion estandar
        desv_std = new_df[item].std()
        # lista para guardar los valores escalados
        valores_esc = []
        for value in new_df[item]:
            # estandariza
            val = ((value - media) / desv_std) 
            # guarda
            valores_esc.append(val)
        # mete toda la lista en la columna
        new_df[item] = valores_esc
    
    return new_df

# NUEVO! <- Estandarizacion Robusta 🦾
# Esto se tiene que llamar una vez que no haya NaNs.
def estandarizacion_robusta(data_frame):
    new_df = data_frame.copy()
    columnas = data_frame.columns.to_list()
    iqr = 0
    for item in columnas[:-1]:
        # media
        iqr = atipicos(new_df[item].to_list())[1]
        # desviacion estandar
        desv_std = new_df[item].std()
        # lista para guardar los valores escalados
        valores_esc = []
        for value in new_df[item]:
            # estandariza
            val = ((value - iqr) / desv_std) 
            # guarda
            valores_esc.append(val)
        # mete toda la lista en la columna
        new_df[item] = valores_esc
    
    return new_df

# =============================================================================================================== #

# ============================================ Atipicos ================================================== #

# NUEVO! <- El anterior hacia cualquier cosa
def atipicos(valores_columna):
    ordered = sorted(valores_columna)
    n = len(valores_columna)
    Q1 = ordered[n // 4]
    Q2 = (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    Q3 = ordered[3 * n // 4]
    iqr = Q3 - Q1
    # print('Max value: ', Q3 + (1.5 * iqr))
    # print('Min value: ', Q1 - (1.5 * iqr))
    # print('\n')
    # Entonces lo que quiero hacer es: Primero identificar los atipicos. Despues buscar esos atipicos en el dataframe, y eliminarlos. NO se hace con el indice xq son indices dis-
    # tintos, la columna no esta ordenada, 'ordered' si.
    values = []
    for value in ordered:
        if ((value > Q3 + (1.5 * iqr)) or (value < Q1 - (1.5 * iqr))):
            values.append(value)
    return values


def limpieza(data_frame):
    new_df = data_frame.copy()
    columnas = new_df.columns.to_list()
    for item in columnas[:-1]:
        indices = []
        valores_at = atipicos(new_df[item].to_list())
        for value in valores_at:
            # Guarda en la lista los indices de las filas que sean iguales al value
            indices.append(new_df[new_df[item] == value].index[0])
        # Y despues las tira todas a la bosta
        new_df = new_df.drop(indices)
    return new_df

# =============================================================================================================== #

# ================================== DIVISION ENTRE POTABLES/NO POTABLES ====================================== #

# Esto esta pensado para sacar las medianas de las columnas cuando el agua es potable/no potable

# Recibe el dataframe, la columna y el valor que te quieras quedar
# Devuelve el nuevo Dataframe (copia modificada, a menos que se pase inplace=true)
def descarte(data_frame, columna, dato):
    new_df = data_frame.copy()
    indexes = new_df[new_df[columna] != dato].index
    return new_df.drop(indexes)

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
    # Aca por las dudas hago lo mismo, igual se lo voy a asignar al original
    new_df = data_frame.copy()
    columnas = new_df.columns.to_list()
    # Esto devuelve:
    # index = indice de la fila
    # fila = todos los datos de la fila en un [[formato]]
    for index, fila in new_df.iterrows():
        # Y por cada fila del dataframe, itero en las columnas
        for i in range(9):
            # Cuando i = 0 > fila['ph'], cuando i = 1 > fila['Hardness']
            if math.isnan(fila.iloc[i]) and fila.iloc[9] == 0:
                new_df.loc[index, columnas[i]] = data_ceros[columnas[i]].median()
            elif math.isnan(fila.iloc[i]) and fila.iloc[9] == 1:
                new_df.loc[index, columnas[i]] = data_unos[columnas[i]].median()
    return new_df

# ================================================================================================================= #



