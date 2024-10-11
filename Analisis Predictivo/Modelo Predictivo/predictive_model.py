import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Cargar datos completos (entrenamiento)
data = pd.read_csv("Analisis Predictivo\\Segmentacion\\data_sin_faltantes.csv")

# Mantener la columna 'SITIO' por separado
sitio = data['SITIO']
data.drop('SITIO', axis=1, inplace=True)

# Seleccionar solo las columnas numéricas
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns

# Crear una copia del conjunto de datos y ocultar un porcentaje de los valores (por ejemplo, 10%)
np.random.seed(0)  # Para reproducibilidad
mask = np.random.rand(*data[numeric_cols].shape) < 0.1  # Ocultar el 10% de los datos
data_with_missing = data[numeric_cols].mask(mask)

# Definir un modelo de KNN
def train_knn(X_train, y_train, n_neighbors=48):
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

# Iterar sobre cada fila del conjunto con valores faltantes
data_imputed = data_with_missing.copy()  # DataFrame para almacenar los valores imputados
for index, row in data_imputed.iterrows():
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
    data_imputed.loc[index] = row

# Evaluar la imputación comparando los valores imputados con los valores originales
mse = mean_squared_error(data[numeric_cols][mask], data_imputed[numeric_cols][mask])
rmse = np.sqrt(mse)
mae = mean_absolute_error(data[numeric_cols][mask], data_imputed[numeric_cols][mask])
r2 = r2_score(data[numeric_cols][mask], data_imputed[numeric_cols][mask])

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Coeficiente de Determinación (R²): {r2}")

# Guardar el DataFrame con los valores imputados en un archivo CSV
data_imputed.insert(0, 'SITIO', sitio)
data_imputed.to_csv('Analisis Predictivo\\Imputacion\\KNN\\data_select_iterative_imputed_KNN.csv', index=False)
