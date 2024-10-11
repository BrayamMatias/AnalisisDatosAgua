import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Cargar datos completos
data = pd.read_csv("Analisis Predictivo\\Segmentacion\\data_sin_faltantes.csv")

# Cargar datos con valores faltantes
data_faltante = pd.read_csv("Analisis Predictivo\\Segmentacion\\data_con_faltantes.csv")

# Mantener la columna 'SITIO' por separado
sitio = data_faltante['SITIO']

# Seleccionar solo las columnas numéricas para la imputación
numeric_data = data.select_dtypes(include=['float64', 'int64'])
numeric_data_faltante = data_faltante.select_dtypes(include=['float64', 'int64'])

# Imputación utilizando Regresión Lineal
data_faltante_imputed = numeric_data_faltante.copy()

for column in numeric_data_faltante.columns:
    if data_faltante[column].isnull().any():
        # Dividir los datos en entrenamiento (sin valores nulos) y prueba (con valores nulos)
        train_data = numeric_data[numeric_data[column].notna()]
        test_data = numeric_data_faltante[numeric_data_faltante[column].isnull()]

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
        data_faltante_imputed.loc[data_faltante[column].isnull(), column] = y_pred

# Convertir valores negativos en positivos y redondear a 4 decimales
data_faltante_imputed = data_faltante_imputed.abs().round(4)

# Agregar la columna 'SITIO' al DataFrame
data_faltante_imputed.insert(0, 'SITIO', sitio)

# Guardar el DataFrame resultante en un archivo CSV
data_faltante_imputed.to_csv('Analisis Predictivo\\Imputacion\\LR\\data_select_imputed_LR.csv', index=False)
