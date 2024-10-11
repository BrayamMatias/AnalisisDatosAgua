import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

# Cargar datos completos (entrenamiento)
data = pd.read_csv("Analisis Predictivo\\Segmentacion\\data_sin_faltantes.csv")

# Cargar datos con valores faltantes (prueba)
data_faltante = pd.read_csv("Analisis Predictivo\\Segmentacion\\data_con_faltantes.csv")

# Mantener la columna 'SITIO' por separado
sitio = data_faltante['SITIO']
data_faltante.drop('SITIO', axis=1, inplace=True)

# Seleccionar solo las columnas numéricas
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns

# Definir un modelo de KNN
def train_knn(X_train, y_train, n_neighbors=48):
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

# Iterar sobre cada fila del conjunto de datos con valores faltantes
for index, row in data_faltante.iterrows():
    # Caso donde no hay valores disponibles en toda la fila
    if row.isna().all():
        # Imputar toda la fila con la media de cada columna
        for col in numeric_cols:
            row[col] = data[col].mean()
    else:
        while row.isna().any():  # Continuar hasta que no haya valores faltantes
            # Determinar qué columnas tienen valores disponibles y cuáles están faltando
            available_cols = row.dropna().index
            missing_cols = row[row.isna()].index

            if len(available_cols) == 0:
                break  # No se puede imputar si no hay valores disponibles

            # Seleccionar la primera columna faltante como objetivo para imputar
            target_col = missing_cols[0]

            # Crear los conjuntos de datos de entrenamiento (sin valores faltantes)
            X_train = data[available_cols]
            y_train = data[target_col]

            # Filtrar el conjunto de entrenamiento eliminando filas con valores faltantes
            X_train = X_train.dropna()
            y_train = y_train.loc[X_train.index]

            # Crear un vector con los valores disponibles de la fila actual
            X_test = row[available_cols].to_frame().T

            # Si no hay suficientes datos en el conjunto de entrenamiento, imputar con la media
            if len(X_train) < 5:
                row[target_col] = data[target_col].mean()
            else:
                # Entrenar el modelo KNN y hacer la predicción para la columna objetivo
                knn_model = train_knn(X_train, y_train)
                predicted_value = knn_model.predict(X_test)[0]

                # Asignar el valor imputado a la columna faltante de la fila actual
                row[target_col] = predicted_value

    # Actualizar el DataFrame con la fila imputada
    data_faltante.loc[index] = row

# Convertir valores negativos en positivos y redondear a 4 decimales
data_faltante = data_faltante.abs().round(4)

# Agregar la columna 'SITIO' nuevamente al DataFrame
data_faltante.insert(0, 'SITIO', sitio)

# Guardar el DataFrame resultante en un archivo CSV
data_faltante.to_csv('Analisis Predictivo\\Imputacion\\KNN\\data_select_imputed_KNN.csv', index=False)
