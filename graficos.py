import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import testing_analisis as ta

df = pd.read_csv('./water_potability.csv', sep=',')
df_ceros = ta.descarte(df, 'Potability', 0)
df_unos = ta.descarte(df, 'Potability', 1)
ta.carga_nans(df, df_ceros, df_unos)
df = ta.normalizacion(df)
df = ta.limpieza_col(df)

def regresion_lin(df, column): # <- Si corren este archivo de una les van a saltar todos los graficos, si los van cerrando aparecen los proximos
    plt.figure(figsize=(8, 5))
    
    # Todas excepto potabilidad
    data = df[[column, 'Potability']]
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
    plt.title(f'{column} contra Potability')
    plt.xlabel('Potability')
    plt.ylabel(column)
    
    plt.show()

# Esto solo itera para printear todas juntas
ploteo_total = df.columns[:-1]
for column in ploteo_total:
    regresion_lin(df, column)

plt.figure(figsize=(10, 8))
matriz_corr = df.corr()

# Colores para la grafica de correlacion -> https://matplotlib.org/stable/users/explain/colors/colormaps.html
sns.heatmap(matriz_corr, annot=True, cmap='inferno', linewidths=0.3)
plt.title('Matriz de Correlacion')
plt.show()