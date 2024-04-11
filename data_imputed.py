from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.impute import IterativeImputer
import pandas as pd
import numpy as np

data = pd.read_csv("data_select_no_faltantes.csv")
data.drop('SITIO', axis=1, inplace=True)
data_faltante_1 = pd.read_csv("data_select_normalizado.csv")
sitio = data_faltante_1['SITIO']
data_faltante_1.drop('SITIO', axis=1, inplace=True)
#data_faltante_1.drop(['TOX_D_48_SUP_UT', 'TOX_D_48_FON_UT', 'TOX_FIS_SUP_15_UT',
 #                     'TOX_FIS_FON_15_UT'], axis=1, inplace=True)
data_faltante = data_faltante_1[data_faltante_1.isnull().any(axis=1)]
data_faltante = data_faltante.replace(' ', np.nan)

# Crear un imputador que usa la regresión lineal para estimar los valores faltantes
imputer = IterativeImputer(estimator=LinearRegression())

# Ajustar el imputador a tus datos completos
imputer.fit(data)

# Transformar tus datos con valores faltantes
data_faltante_imputed = imputer.transform(data_faltante)

# 'data_faltante_imputed' es un array
data_faltante_imputed = pd.DataFrame(
    data_faltante_imputed, columns=data_faltante.columns)

#Convertir negativos en positivos
data_faltante_imputed = data_faltante_imputed.abs()

# Redondear a 4 decimales
data_faltante_imputed = data_faltante_imputed.round(4)

# Agregar la columna 'SITIO' al DataFrame
data_faltante_imputed.insert(0, 'SITIO', sitio)

# Guardar el DataFrame en un archivo CSV
data_faltante_imputed.to_csv('data_select_imputed.csv', index=False)
# Ahora, 'data_faltante_imputed' es tu DataFrame original, pero con los valores faltantes rellenados con las predicciones del imputador
