import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Caracteristicas: SITIO	DBO_mg/L	DQO_mg/L	SST_mg/L	COLI_FEC_NMP_100mL	E_COLI_NMP_100mL OD_PORC	TOX_D_48_UT	TOX_V_15_UT

# Carga el DataFrame con valores faltantes
data = pd.read_excel("Datos seleccionados.xlsx")
# Elimina las columnas que no se van a utilizar
data.drop(['TOX_D_48_SUP_UT', 'TOX_D_48_FON_UT', 'TOX_FIS_SUP_15_UT',
          'TOX_FIS_FON_15_UT'], axis=1, inplace=True)

def label_encoder(datas, data_category):
    # Verifica si la columna es de tipo 'object' (categórica)
    if datas[data_category].dtype == 'object':
        le = LabelEncoder()
        # Ajusta el LabelEncoder a todos los valores de string únicos en la columna
        le.fit(datas[data_category].unique())
        # Aplica la transformación a toda la columna
        datas[data_category] = le.transform(datas[data_category])

# Lista de variables categóricas
variables = ['SITIO', 'DBO_mg/L', 'DQO_mg/L', 'SST_mg/L', 'COLI_FEC_NMP_100mL',
             'E_COLI_NMP_100mL', 'OD_PORC', 'TOX_D_48_UT', 'TOX_V_15_UT']
for var in variables:
    label_encoder(data, var)

# Guarda el DataFrame normalizado
data.to_csv('data_select_normalizado_LE.csv', index=False)

data = pd.read_csv("data_select_normalizado_LE.csv")

# Guarda el DataFrame con valores faltantes en un archivo
data_con_faltantes = data[data.isnull().any(axis=1)]
data_con_faltantes.to_csv('data_select_con_faltantes_LE.csv', index=False)

# Guarda el Dataframe sin valores faltantes en un archivo
data_sin_faltantes = data.dropna()
data_sin_faltantes.to_csv('data_select_sin_faltantes_LE.csv', index=False)
