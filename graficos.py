import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import testing_analisis as ta

'''
    Hacer la relacion de Pearson

'''

df_csv = pd.read_csv('./water_potability.csv', sep=',')
df_csv.fillna(df_csv.median(), inplace=True)
df = df_csv.copy()
df_ceros = ta.descarte(df, 'Potability', 0)
df_unos = ta.descarte(df, 'Potability', 1)
ta.carga_nans(df, df_ceros, df_unos)
df = ta.estandarizacion(df)
df = ta.normalizacion(df)
df = ta.limpieza_col(df)

# ================================== GRAFICOS CON EL DATA FRAME SIN TRATAR ========================================== #

# Graficos de dispersion de los datos
def scattered(df, last_col):
    columns = df.columns.to_list()
    for item in columns[:last_col]:
        # Esta rara la regresion esta, es como si estuviera recta en el 0.5 <-
        # fit = LinearRegression().fit(df.index.values.reshape(-1, 1), df[item])
        # m = fit.coef_.flatten()
        # b = fit.intercept_.flatten()
        # plt.plot(df.index, m*df[item]+b, color='red')
        plt.xlabel('Index')
        plt.ylabel(f'Column \'{item}\'')
        plt.xticks(rotation=45, horizontalalignment='center')
        plt.minorticks_on()
        plt.grid()
        plt.title('PLOT PER COLUMN')
        plt.scatter(df.index, df[item], label=item.upper(), color='slateblue', marker='.')
        plt.legend()
        plt.show()

# Histogramas de los datos
def histograms(df, last_col):
    columns = df.columns.to_list()
    for item in columns[:last_col]:
        plt.xlabel(f'Column \'{item}\'')
        # Aca por que aparece hasta el 700? <-
        plt.ylabel('Index')
        plt.xticks(rotation=45, horizontalalignment='center')
        plt.minorticks_on()
        plt.grid()
        plt.title('PLOT PER COLUMN')
        plt.hist(df[item], bins=15, color='slateblue', edgecolor='black', label=item.upper())
        plt.legend()
        plt.show()

# Matriz de correlacion
def matrix_corr(df):
    plt.figure(figsize=(10, 8))
    matriz_corr = df.corr()
    # Colores para la grafica de correlacion -> https://matplotlib.org/stable/users/explain/colors/colormaps.html
    sns.heatmap(matriz_corr, annot=True, cmap='inferno', linewidths=0.3)
    plt.title('Correlation Matrix')
    plt.show()

# ================================================================================================================= #

