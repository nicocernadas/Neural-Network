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

# ============================== Limpieza de Atipicos ================================== #

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




# ================================ DATA RANDOM =========================================
# DataFrame con valores Null
# ds = pd.read_csv('water_potability.csv', sep=',')
# print(ds.describe())
# Vistaso del DataFrame
# np_array = ds.to_numpy()
# columns = ["PH","HARDNESS","SOLIDS","CHLORAMINES","SULFATE","CONDUCTIVITY", "ORGANIC_CARBON","TRIHALOMETHANES", "TURBIDITY", "POTABILITY"]
# print("{:<18} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18}".format(*columns))
# print("-" * 76)
# for row in np_array[:10,]:
#     print("{:<18} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18}".format(*row))
# Para que printee todas las columnas sin los ' . . '

# =================================== COMANDOS UTILES ===================================
# Operaciones con matrices elemento por elemento
# np.add(a,b), np.subtract(a,b), np.multiply(a,b), np.devide(a,b), np.sqrt(a)
# Operaciones Matriciales
# np.dot(a,b) => producto punto
# a @ b => producto punto
# np.sum(a, (opcional axis = x)) => suma de todos los elementos (opcional en un determinado eje)
# .T o transpose() para transponer una matriz
# Matriz identidad
# identity = np.eye(6,6)
# print(identity)

# print('Basicos')
# print(df.index)
# print('\n')

# print('Columns')
# print(df.columns)
# print('\n')

# columns = df.columns
# del df['Data Usage (MB/day)'], df['Battery Drain (mAh/day)'], df['User ID']
# print(columns)
# print('\n')

# print('Matriz de series con los datos (1 fila por cada fila del dataframe)')
# print(df.values)
# print('\n')

# print('Describe')
# print(df.describe())
# print('\n')

# print('.at[fila, columna_nombre]')
# print(df.at[0,'App Usage Time (min/day)'])
# print('\n')

# print('df[\'nombre_col\'][primer_index_fil: ultimo_index_fil]')
# print(df['ph'][:10])

# print('.iat[fila, columna_index]')
# print(df.iat[0,2])
# print('\n')

# print('fila completa')
# print(df.loc[0])
# print('\n')

# print('asi devuelve un dataframe')
# print(df.iloc[[0]])
# print('\n')

# print('asi devuelve una serie')
# print(df.iloc[0])
# print('\n')

# funcion .groupby() mirar bien xq tiene banda de usos
# print(df.groupby('Solids').size())

# Para insertar una lista de datos en una determinada posicion
# np_array = np.insert(np_array, 0, [[columns]], axis=0)

# pd.read_csv("nombre_csv", index_col=0)
# index_col= x => donde x es el numero de columna que tomara el dataframe como indice de la izquierda (son validos tanto un numero o un ['nombre'])
# tambien podriamos hacer df.set_index('nombre_col') para lo mismo

# low_ph = df[ df['ph'] <= 3] #tambien se puede seleccionar asi: todos los datos de la columna ph que tengan valores menores o iguales a 3
# print(low_ph)

# para ordenar los datos en base a una sola columna
# df.sort_values(by='nombre_col')

#=======================================================================================