import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

df = pd.read_csv('water_potability.csv', sep=',')

def regresion_lin(df, column):
    plt.figure(figsize=(8, 5))
    
    # Todas excepto potabilidad
    data = df[[column, 'Potability']].dropna()
    X = data['Potability'].values.reshape(-1, 1)
    y = data[column].values
    
    # Arma la regresion
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Puntos
    plt.scatter(X, y, color='blue', label='Data')
    
    # Regresion lineal
    plt.plot(X, y_pred, color='red', label='Linear Regression')
    
    # Titulos y Ejes
    plt.title(f'{column} and Potability')
    plt.xlabel('Potability')
    plt.ylabel(column)
    
    plt.show()

# Esto solo itera para printear todas juntas
ploteo_total = df.columns[:-1]
for column in ploteo_total:
    regresion_lin(df, column)


plt.figure(figsize=(10, 8))
matriz_corr = df.corr()

sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', linewidths=0.3)
plt.title('Matriz de Correlacion')
plt.show()