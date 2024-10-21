import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import testing_analisis as ta

df = pd.read_csv('./water_potability.csv', sep=',')
df_ceros = ta.descarte(df, 'Potability', 0)
df_unos = ta.descarte(df, 'Potability', 1)
ta.carga_nans(df, df_ceros, df_unos)
df = ta.normalizacion(df)
df = ta.limpieza_col(df)

X = df.values[:, :-1]
Y = df.values[:, -1]

# Separo entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3)

nn = MLPClassifier(
    solver='sgd', # Otra metodologia de entrenamiento          
    hidden_layer_sizes=(9,),
    activation='relu',
    max_iter=200_000,
    learning_rate_init=.001,
    # random_state=42 # Semilla. Siempre pone los mismos pesos al principio, y divide la red igual. Es para mejor visualizacion
)

nn.fit(X_train, Y_train) # <- esto no entiendo bien que es

# Despues de la limpieza y todo, esto sigue igual.
# No pasan del 60-62%
print("Puntaje del conjunto de entrenamiento: %f" % nn.score(X_train, Y_train))
print("Puntaje del conjunto de prueba: %f" % nn.score(X_test, Y_test))