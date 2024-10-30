import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

df = pd.read_csv('./csvs/transf_data.csv', sep=',')

# ============================================== SCRIPT DE FUNCIONES ===================================================== #

# Funcion de descarte
# Asigna a un nuevo dataframe, otro pero con los indices dropeados de las filas que cumplan la condicion pasada
# Ejemplo: Si quiero armar un nuevo dataframe sin las filas donde la potabilidad es 0:
# df_1 = descarte(df, 'Potability', 1) => Esto se guarda todas las filas donde Potabilidad es 1
def descarte(data_frame, columna, dato):
    new_df = data_frame.copy()
    indexes = new_df[new_df[columna] != dato].index
    return new_df.drop(indexes)

# Cargadora de NaNs
def carga_nans(data_frame, data_ceros, data_unos):
    # Aca por las dudas hago lo mismo, igual se lo voy a asignar al original
    new_df = data_frame.copy()
    columnas = new_df.columns.to_list()
    # Esto devuelve:
    # index = indice de la fila
    # fila = todos los datos de la fila en un [[formato]]
    for index, fila in new_df.iterrows():
        # Y por cada fila del dataframe, itero en las columnas
        for i in range(9):
            # Cuando i = 0 > fila['ph'], cuando i = 1 > fila['Hardness']
            if math.isnan(fila.iloc[i]) and fila.iloc[9] == 0:
                new_df.loc[index, columnas[i]] = data_ceros[columnas[i]].median()
            elif math.isnan(fila.iloc[i]) and fila.iloc[9] == 1:
                new_df.loc[index, columnas[i]] = data_unos[columnas[i]].median()
    return new_df

# Normalizacion
def normalizacion(data_frame):
    # Se agrego esto, sino se esta modificando el dataframe original, y no quiero eso. Se modifica en todas las funciones
    new_df = data_frame.copy() # <-
    columnas = new_df.columns.to_list()
    new_val = []
    for item in columnas[:-1]:
        max_value = new_df[item].max()
        for value in new_df[item]:   
            new_val.append(value/max_value)
        new_df[item] = new_val
        new_val.clear()
    return new_df

# Borrador de valores atipicos
def atipicos(valores_columna):
    ordered = sorted(valores_columna)
    n = len(valores_columna)
    Q1 = ordered[n // 4]
    Q2 = (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    Q3 = ordered[3 * n // 4]
    iqr = Q3 - Q1
    # print('Max value: ', Q3 + (1.5 * iqr))
    # print('Min value: ', Q1 - (1.5 * iqr))
    # print('\n')
    # Entonces lo que quiero hacer es: Primero identificar los atipicos. Despues buscar esos atipicos 
    # en el dataframe, y eliminarlos. NO se hace con el indice xq son indices dis-
    # tintos, la columna no esta ordenada, 'ordered' si.
    values = []
    for value in ordered:
        if ((value > Q3 + (1.5 * iqr)) or (value < Q1 - (1.5 * iqr))):
            values.append(value)
    return values, iqr

def limpieza(data_frame):
    new_df = data_frame.copy()
    columnas = new_df.columns.to_list()
    for item in columnas[:-1]:
        indices = []
        valores_at = atipicos(new_df[item].to_list())[0]
        for value in valores_at:
            # Guarda en la lista los indices de las filas que sean iguales al value
            indices.append(new_df[new_df[item] == value].index[0])
        # Y despues las tira todas a la bosta
        new_df = new_df.drop(indices)
    return new_df

# Estandarizacion
def estandarizacion(data_frame):
    new_df = data_frame.copy()
    columnas = data_frame.columns.to_list()
    # Va a guardar las medias y desv de todas las columnas, se va a usar mas tarde
    med_lst = []
    std_lst = []
    for item in columnas[:-1]:
        # media
        media = new_df[item].mean()
        med_lst.append(media)
        # desviacion estandar
        desv_std = new_df[item].std()
        std_lst.append(desv_std)
        # lista para guardar los valores escalados
        valores_esc = []
        for value in new_df[item]:
            # estandariza
            val = ((value - media) / desv_std) 
            # guarda
            valores_esc.append(val)
        # mete toda la lista en la columna
        new_df[item] = valores_esc
    return new_df, med_lst, std_lst

def estandarizacion_muestras(data_frame, medias, std_dev):
    # A esto hay que pasarle las listas con medias y desv
    new_df = data_frame.copy()
    columnas = data_frame.columns.to_list()
    i = 0
    for item in columnas:
        val = new_df[item][0]
        val_new = ((val - medias[i]) / std_dev[i])
        new_df[item][0] = val_new
    return new_df


# Estandarizacion Robusta
# Esto se tiene que llamar una vez que no haya NaNs.
def estandarizacion_robusta(data_frame):
    new_df = data_frame.copy()
    columnas = data_frame.columns.to_list()
    iqr = 0
    for item in columnas[:-1]:
        # media
        iqr = atipicos(new_df[item].to_list())[1]
        # desviacion estandar
        desv_std = new_df[item].std()
        # lista para guardar los valores escalados
        valores_esc = []
        for value in new_df[item]:
            # estandariza
            val = ((value - iqr) / desv_std)
            # guarda
            valores_esc.append(val)
        # mete toda la lista en la columna
        new_df[item] = valores_esc
    
    return new_df

def borrador_samples(data_frame):
    new_df = data_frame.copy()
    indices = []
    for index,value in new_df.iterrows():
        if new_df['fraud'][index] == 0:
            indices.append(index)
        if (index == 820_000):
            newnew_df = new_df.drop(indices)
            return newnew_df
        
# ============================================================================================================================ #

# ============================================== GRAFICOS ===================================================== #

# Graficos de dispersion de los datos
def scattered(df, last_col):
    columns = df.columns.to_list()
    for item in columns[:last_col]:
        plt.ylabel(f'Column \'{item}\'')
        plt.xticks(rotation=45, horizontalalignment='center')
        plt.minorticks_on()
        plt.grid()
        plt.title('PLOT PER COLUMN')
        plt.scatter(df.index, df[item], label=item.upper(), color='slateblue', marker='.')
        plt.legend()
        plt.show()

# Graficos de distribucion (boxplot)
def boxplot(data_frame, last_col):
    columnas = data_frame.columns.to_list()
    fig, ax = plt.subplots(last_col, 1, figsize=(10, 20))
    fig.subplots_adjust(hspace=0.75)
    for i in range(last_col):
        sns.boxplot(x=data_frame[columnas[i]], data=data_frame, ax=ax[i])
        plt.show()

# Todas las columnas en funcion de una
def catplot(data_frame, last_col=1, hue=None):
    if hue == None:
        return 'No se especifico una columna de comparacion. => hue=\'{columna en funcion de la que se quiera graficar}\''
    columns = data_frame.columns.to_list()
    del columns[-1]
    plt.figure(figsize=(10,5))
    for item in columns[:last_col]:
        sns.catplot(x = data_frame[hue], y = data_frame[item], data=data_frame, palette='crest')
    plt.show()

# Histogramas de los datos
def histograms(df, last_col):
    new_df = df.copy()
    columns = new_df.columns.to_list()
    for item in columns[:last_col]:
        
        media = new_df[item].mean()
        desv_est = new_df[item].std()
        array = new_df[item].to_numpy()

        plt.xlabel(f'Column \'{item}\'')
        plt.ylabel('Index')
        plt.xticks(rotation=45, horizontalalignment='center')
        plt.minorticks_on()
        plt.grid()
        plt.title('PLOT PER COLUMN')

        cont, bins, patches = plt.hist(array, color='slateblue', edgecolor='black', density=True, alpha=0.6, label=item.upper())
        x_densidad = np.linspace(bins[0], bins[-1], 100)

        funcion = 1/(desv_est*np.sqrt(2*np.pi)) * np.exp(-0.5*((x_densidad - media)/desv_est)**2)

        plt.plot(x_densidad, funcion, linewidth=2, color='black')
        plt.legend()
        plt.show()

# Matriz de correlacion
def matrix_corr(df):
    plt.figure(figsize=(10, 8))
    matriz_corr = df.corr()
    # Colores para la grafica de correlacion -> https://matplotlib.org/stable/users/explain/colors/colormaps.html
    sns.heatmap(matriz_corr, annot=True, cmap='inferno', linewidths=0.3, vmin=-1)
    plt.xticks(horizontalalignment='center')
    plt.title('Correlation Matrix')
    plt.show()

# Pares de graficos en funciones del resto
def pplot(data_frame, hue):
    sns.pairplot(data_frame, hue=hue, corner=True, palette='plasma')
    plt.show()

# Grafico del entrenamiento de la red (test/train)
def grafico_acc(L, train_l, test_l):
    fmt_train = {
        'color': 'tab:red',
        'ls': 'solid',
        'lw': 3,
    }

    fmt_test = {
        'color': 'tab:orange',
        'ls': 'solid',
        'lw': 3,
    }

    fig, (ax) = plt.subplots(1, 1, figsize=(10,8))

    ax.plot(train_l, label='Train', **fmt_train)
    ax.plot(test_l, label='Test', **fmt_test)

    ax.grid(which='both')
    ax.legend()
    ax.set_title(f'Accuracy {L=}')
    ax.set_xlabel('Step')

    fig.tight_layout()
    plt.show()

# ============================================================================================================= #