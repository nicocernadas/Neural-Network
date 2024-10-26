import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from script_funciones import estandarizacion, limpieza

df = pd.read_csv('transf_data.csv', sep=',')

df = limpieza(df)
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
    learning_rate_init=.03,
    random_state=42 # Semilla. Siempre pone los mismos pesos al principio, y divide la red igual. Es para mejor visualizacion
)

nn.fit(X_train, Y_train) # <- esto no se bien que es

print("Puntaje del conjunto de entrenamiento: %f" % nn.score(X_train, Y_train))
print("Puntaje del conjunto de prueba: %f" % nn.score(X_test, Y_test))