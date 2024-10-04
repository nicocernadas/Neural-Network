import numpy as np
import pandas as pd #en las consignas dice solo numpy, pero me imagino q para cargar el df se puede usar pandas

#https://pandas.pydata.org/docs/reference/frame.html
#https://numpy.org/doc/stable/user/index.html << https://joserzapata.github.io/courses/python-ciencia-datos/numpy/


df = pd.read_csv('user_behavior_dataset.csv', sep=',')
print(df.columns)
del df['Data Usage (MB/day)'], df['Battery Drain (mAh/day)'], df['User ID'], df['Device Model'], df['Operating System']
# df to numpy array
# No incluye las columnas en la extraccion
# cada fila del array es una lista con los datos de cada columna
np_array = df.to_numpy()

# Vistaso del DataFrame
columns = ["M PER DAY","SCREEN on H/D","APPS","AGE","GENDER","CLASS (1-5)"]
print("{:<10} | {:<9} | {:<10} | {:<7} | {:<10} | {:<10}".format(*columns))
print("-" * 76)
for row in np_array[:5,]:
    print("{:<10} | {:<13} | {:<10} | {:<7} | {:<10} | {:<10}".format(*row))

i = 0
for gender in np_array[:,4]:
    if gender == 'Male':
        np_array[i,4] = 1
        i += 1
    else:
        np_array[i,4] = 2
        i += 1
        
# Con los generos intercambiados
print("{:<10} | {:<9} | {:<10} | {:<7} | {:<10} | {:<10}".format(*columns))
print("-" * 76)
for row in np_array[:5,]:
    print("{:<10} | {:<13} | {:<10} | {:<7} | {:<10} | {:<10}".format(*row))























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
# #df basics
# print('Basics')
# print(df.index)
# print('\n')
# #deleting non usable columns
# print('Columns')
# print(df.columns)
# print('\n')
# print('Erasing Columns')
# del df['Data Usage (MB/day)'], df['Battery Drain (mAh/day)'], df['User ID']
# columns = df.columns
# print(columns)
# print('\n')
# #matrix of values (usable for numpy)
# print('Matrix of values')
# print(df.values)
# print('\n')

# #Description of the Data
# print('Describe')
# print(df.describe())
# print('\n')
# #Accessing a singular position .at[row, column_name]
# print('.at[row, column_name]')
# print(df.at[0,'App Usage Time (min/day)'])
# print('\n')
# #Accessing a singular position .iat[row, column_index]
# print('.iat[row, column_index]')
# print(df.iat[0,2])
# print('\n')
# #Accesing whole row
# print('whole row')
# print(df.loc[0])
# print('\n')
# #Accessing by purely integer position
# #This is a DataFrame
# print('iloc dataframe')
# print(df.iloc[[0]])
# print('\n')
# #This is a Series
# print('iloc series')
# print(df.iloc[0])
# print('\n')
# Para insertar una lista de datos en una determinada posicion
# np_array = np.insert(np_array, 0, [[columns]], axis=0)
#=======================================================================================