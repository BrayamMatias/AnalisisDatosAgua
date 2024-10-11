import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Supongamos que 'data_sin_faltantes' es el conjunto original con datos completos
data_sin_faltantes = pd.read_csv("Analisis Predictivo\\Segmentacion\\data_sin_faltantes.csv")
data = data_sin_faltantes.copy()

#seleccionar solo las columnas numéricas para la imputación
data = data.select_dtypes(include=['float64', 'int64'])

# Crear un conjunto con valores faltantes simulados (ocultar un porcentaje de los datos)
np.random.seed(0)  # Para reproducibilidad
mask = np.random.rand(*data.shape) < 0.1  # Ocultar el 10% de los datos
data_with_missing = data.mask(mask)

# Imputar los datos faltantes utilizando el imputador entrenado
imputer = KNNImputer(n_neighbors=48)
data_imputed = imputer.fit_transform(data_with_missing)

# Convertir el array imputado a DataFrame
data_imputed = pd.DataFrame(data_imputed, columns=data.columns)

# Evaluar la imputación comparando los datos imputados con los valores originales
mse = mean_squared_error(data[mask], data_imputed[mask])
rmse = np.sqrt(mse)
mae = mean_absolute_error(data[mask], data_imputed[mask])
r2 = r2_score(data[mask], data_imputed[mask])

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Coeficiente de Determinación (R²): {r2}")
