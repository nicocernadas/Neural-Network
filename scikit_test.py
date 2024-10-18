import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
# trae limpio
# from testing_analisis import df
# importa aca mismo
df = pd.read_csv('./water_potability.csv', sep=',')
df.fillna(df.median(), inplace=True)

X = df.values[:, :-1]
Y = df.values[:, -1]

# Separo entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3)

nn = MLPClassifier(
    solver='sgd', # Otra metodologia de entrenamiento          
    hidden_layer_sizes=(9,),
    activation='relu',
    max_iter=200_000,
    learning_rate_init=.01,
    # random_state=42 # Semilla. Siempre pone los mismos pesos al principio, y divide la red igual. Es para mejor visualizacion
)

nn.fit(X_train, Y_train)

print("Puntaje del conjunto de entrenamiento: %f" % nn.score(X_train, Y_train))
print("Puntaje del conjunto de prueba: %f" % nn.score(X_test, Y_test))