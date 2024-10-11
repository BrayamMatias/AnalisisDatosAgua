from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np

# Cargar datos completos
data = pd.read_csv("data_select_sin_faltantes_MMS.csv")
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Cargar datos con valores faltantes
data_faltante = pd.read_csv("data_select_con_faltantes_MMS.csv")
numeric_data_faltante = data_faltante.select_dtypes(include=['float64', 'int64'])


# Reemplazar valores de espacio en blanco por NaN
numeric_data_faltante = data_faltante.replace(' ', np.nan)

# Crear un imputador que use KNN para estimar los valores faltantes
imputer = KNNImputer(n_neighbors=5)

# Ajustar el imputador solo a los datos completos
imputer.fit(data)

# Transformar los datos con valores faltantes
data_faltante_imputed = imputer.transform(numeric_data_faltante)

# 'data_faltante_imputed' es un array, convertirlo en DataFrame
data_faltante_imputed = pd.DataFrame(data_faltante_imputed, columns=numeric_data_faltante.columns)

# Convertir valores negativos en positivos
data_faltante_imputed = data_faltante_imputed.abs()

# Redondear a 4 decimales
data_faltante_imputed = data_faltante_imputed.round(4)

# Agregar la columna 'SITIO' al DataFrame
data_faltante_imputed.insert(0, 'SITIO', sitio)

# Guardar el DataFrame en un archivo CSV
data_faltante_imputed.to_csv('data_select_imputed_KNN_MMS.csv', index=False)
