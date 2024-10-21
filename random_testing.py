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
# accuracy test ========================================================================
# from sklearn.model_selection import train_test_split

# # Extraigo las columnas de entrada
# inputs = df.iloc[:, 0:9].values

# outputs = df.iloc[:, -1].values

# # Conjuntos de entrenamiento y prueba
# x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size=1/3)

# #shape retorna una tupla con las dimensiones de la matriz = (filas, columnas).
# # por lo que shape[0], nos retorna las filas de la matriz.
# n = x_train.shape[0] # número de registros de entrenamiento

# # Red neuronal
# # Capa oculta 1 (9 entradas, 9 neuronas)
# w_hidden_1 = np.random.rand(9, 9)
# print(w_hidden_1)

# # Sesgo
# b_hidden_1 = np.random.rand(9, 1)

# # Capa de salida
# w_output = np.random.rand(1, 9)
# # Sesgo
# b_output = np.random.rand(1, 1)

# # Funciones de Activacion
# relu = lambda x: np.maximum(x, 0)
# sigmoide = lambda x: 1 / (1 + np.exp(-x))

# def f_prop(X):
#     # Capa oculta
#     z1 = w_hidden_1 @ X + b_hidden_1
#     a1 = relu(z1)
    
#     # Capa de salida
#     z2 = w_output @ a1 + b_output 
#     a2 = sigmoide(z2)

#     return z1, a1, z2, a2

# test_predictions = f_prop(x_test.transpose())[3]
# test_comparisons = np.equal((test_predictions >= .5).flatten().astype(int), y_test)
# accuracy = sum(test_comparisons.astype(int) / x_test.shape[0])

# # muy poca accuracy, aprox 40%
# print("ACCURACY: ", accuracy)
# ======================================================================================
# graficos =============================================================================
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression

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
# =============================================================================================
# error cuadrado ==============================================================================
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
# =============================================================================================
# limpieza de atipicos ========================================================================
# def atipicos_col(values_list):
#     sample_ordenado = sorted(values_list)
#     n = len(sample_ordenado)

#     Q1 = sample_ordenado[n // 4]
#     Q2 = (sample_ordenado[n // 2 - 1] + sample_ordenado[n // 2]) / 2
#     Q3 = sample_ordenado[3 * n // 4]
#     iqr = Q3 - Q1

#     # Esto ahora cambio un poco, ya no guardo los atipicos, se van a guardar las posiciones
#     indices_atipicos = []

#     # Esto retorna por cada lista, el indice (index) y el valor (x)
#     for index, x in enumerate(sample_ordenado):
#         if (x > Q3 + (1.5 * iqr) or (x < Q1 - (1.5 * iqr))):
#             indices_atipicos.append(index)

#     return indices_atipicos

# def limpieza_col(data_frame):
#     columnas = data_frame.columns.to_list()

#     for item in columnas[:-1]:
#         # Ahora se pasa cada columna como una lista
#         indices_atipicos = atipicos_col(data_frame[item].to_list())
        
#         # Si no esta vacia...
#         if indices_atipicos:
#             data_frame = data_frame.drop(data_frame.index[indices_atipicos])

#     return data_frame
# ==============================================================================================