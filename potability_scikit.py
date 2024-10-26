import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from script_funciones import carga_nans, descarte, estandarizacion, limpieza

df = pd.read_csv('./water_potability.csv', sep=',')

columnas = df.columns.to_list()
df_ceros = descarte(df, 'Potability', 0)
df_unos = descarte(df, 'Potability', 1)
df = carga_nans(df, df_ceros, df_unos)
df = estandarizacion(df)

# print(df)
X = df.values[:, :-1]
Y = df.values[:, -1]

# Separo entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3)

nn = MLPClassifier(
    solver='sgd',          
    hidden_layer_sizes=(4,),
    activation='relu',
    max_iter=15_000,
    learning_rate_init=.01,
    random_state=42 # Semilla. Siempre pone los mismos pesos al principio, y divide la red igual. Es para mejor visualizacion
)

nn.fit(X_train, Y_train) # <- esto no se bien que es

print("Puntaje del conjunto de entrenamiento: %f" % nn.score(X_train, Y_train))
print("Puntaje del conjunto de prueba: %f" % nn.score(X_test, Y_test))


'''
============================= FUNCIONES DE SCIKIT NO UTILIZADAS =====================================
                                     (pero si testeadas)

MINMAX
Agarra el valor minimo, y pasa a ser el cero, hace lo mismo con el max.
Escala los datos en funcion de eso (array de numpy)
min_max = preprocessing.MinMaxScaler().fit_transform(df)

NORMALIZER
Normalizados = X / sqrt( X1^2 + X2^2 + X3^2 ...) <- Esto es lo que estaba haciendo en un principio 
(dividir por la media (xq esto es eso, la suma total de los valores al cuadrado y despues raiz))
normalizer = preprocessing.Normalizer().transform(df.T)
normalizer = normalizer.T

ESTANDARIZADOR Y ESTANDARIZADOR ROBUSTO
Estandarizacion media=0, dsv estandar=1
estandarizados = preprocessing.StandardScaler().fit_transform(df)

Este es otro tipo de estandarizacion => 
estandarizados_rob = preprocessing.RobustScaler().fit_transform(df)
El segundo hace que los datos esten menos distribuidos (mas cerca del cero)
df = pd.DataFrame(estandarizados, columns=columnas)

print(df['Potability'][3275])

for index, values in df.iterrows():
    if values[9] == -0.7997747430618446:
        df.loc[index, 'Potability'] = 0
    else:
        df.loc[index, 'Potability'] = 1

========================================================================================================
'''