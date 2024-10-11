import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Cargar datos sin faltantes
data_sin_faltantes = pd.read_csv("Analisis Predictivo\\Segmentacion\\data_sin_faltantes.csv")
data = data_sin_faltantes.copy()

# Seleccionar solo las columnas numéricas para la imputación
data = data.select_dtypes(include=['float64', 'int64'])

# Crear un conjunto con valores faltantes simulados (ocultar un porcentaje de los datos)
np.random.seed(0)  # Para reproducibilidad
mask = np.random.rand(*data.shape) < 0.1  # Ocultar el 10% de los datos
data_with_missing = data.mask(mask)

# Imputación utilizando Regresión Lineal
data_imputed = data_with_missing.copy()

for column in data.columns:
    if data_with_missing[column].isnull().any():
        # Dividir los datos en entrenamiento (sin valores nulos) y prueba (con valores nulos)
        train_data = data_with_missing[data_with_missing[column].notna()]
        test_data = data_with_missing[data_with_missing[column].isnull()]

        # Definir las características (todas las columnas excepto la que se va a predecir)
        X_train = train_data.drop(column, axis=1)
        y_train = train_data[column]

        X_test = test_data.drop(column, axis=1)

        # Imputar valores faltantes en X_train y X_test usando la media
        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # Entrenar el modelo de regresión lineal
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predecir los valores faltantes
        y_pred = model.predict(X_test)

        # Rellenar los valores faltantes con las predicciones
        data_imputed.loc[data_with_missing[column].isnull(), column] = y_pred

# Evaluar la imputación comparando los datos imputados con los valores originales
mse = mean_squared_error(data[mask], data_imputed[mask])
rmse = np.sqrt(mse)
mae = mean_absolute_error(data[mask], data_imputed[mask])
r2 = r2_score(data[mask], data_imputed[mask])

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Coeficiente de Determinación (R²): {r2}")
