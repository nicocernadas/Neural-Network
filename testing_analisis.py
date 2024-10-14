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
            # guarda
            valores_esc.append(valor_esc)
        # mete toda la lista en la columna
        data_frame[item] = valores_esc
    
    return data_frame

df = estandarizacion(df)

# =============================================================================================================== #


