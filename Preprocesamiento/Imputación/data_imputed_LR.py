from sklearn.linear_model import LinearRegression
from sklearn.impute import IterativeImputer
import pandas as pd
import numpy as np

# Cargar datos completos
data = pd.read_csv("data_select_sin_faltantes.csv")

# Eliminar la columna 'SITIO'
data.drop('SITIO', axis=1, inplace=True)

# Cargar datos con valores faltantes
data_faltante = pd.read_csv("data_select_con_faltantes.csv")

# Extraer la columna 'SITIO'
sitio = data_faltante['SITIO']

# Eliminar la columna 'SITIO'
data_faltante.drop('SITIO', axis=1, inplace=True)

# Reemplazar valores de espacio en blanco por NaN
data_faltante = data_faltante.replace(' ', np.nan)

# Crear un imputador que use regresión lineal para estimar los valores faltantes
imputer = IterativeImputer(estimator=LinearRegression())

# Ajustar el imputador solo a los datos completos
imputer.fit(data)

# Transformar los datos con valores faltantes
data_faltante_imputed = imputer.transform(data_faltante)

# 'data_faltante_imputed' es un array, convertirlo en DataFrame
data_faltante_imputed = pd.DataFrame(data_faltante_imputed, columns=data_faltante.columns)

# Convertir valores negativos en positivos
data_faltante_imputed = data_faltante_imputed.abs()

# Redondear a 4 decimales
data_faltante_imputed = data_faltante_imputed.round(4)

# Agregar la columna 'SITIO' al DataFrame
data_faltante_imputed.insert(0, 'SITIO', sitio)

# Guardar el DataFrame en un archivo CSV
data_faltante_imputed.to_csv('data_select_imputed_LR.csv', index=False)
