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