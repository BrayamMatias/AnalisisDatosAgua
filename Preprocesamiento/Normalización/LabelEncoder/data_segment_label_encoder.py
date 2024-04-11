import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Caracteristicas: SITIO	DBO_mg/L	DQO_mg/L	SST_mg/L	COLI_FEC_NMP_100mL	E_COLI_NMP_100mL OD_PORC	TOX_D_48_UT	TOX_V_15_UT

# Carga el DataFrame con valores faltantes
data = pd.read_excel("Datos seleccionados.xlsx")
# Elimina las columnas que no se van a utilizar
data.drop(['TOX_D_48_SUP_UT', 'TOX_D_48_FON_UT', 'TOX_FIS_SUP_15_UT',
          'TOX_FIS_FON_15_UT'], axis=1, inplace=True)

# convierte las variables categóricas a numéricas
# data = pd.read_csv("data_select_no_faltantes.csv")


def label_encoder(datas, data_category):
    # Verifica si la columna es de tipo 'object' (categórica)
    if datas[data_category].dtype == 'object':
        # Convierte los valores que comienzan con '<' a números y agrega un valor constante
        datas[data_category] = datas[data_category].apply(lambda x: float(
            x[1:]) if type(x) == str and x.startswith('<') else x)
        le = LabelEncoder()
        # Ajusta el LabelEncoder a todos los valores de string únicos en la columna que no comienzan con '<'
        le.fit(datas[data_category][datas[data_category].apply(
            lambda x: type(x) == str and not x.startswith('<'))].unique())
        # Aplica la transformación solo a las celdas que son strings y no comienzan con '<'
        datas[data_category] = datas[data_category].apply(lambda x: le.transform(
            [x])[0] if type(x) == str and not x.startswith('<') else x)


# Lista de variables categóricas
variables = ['SITIO', 'DBO_mg/L', 'DQO_mg/L', 'SST_mg/L', 'COLI_FEC_NMP_100mL',
             'E_COLI_NMP_100mL', 'OD_PORC', 'TOX_D_48_UT', 'TOX_V_15_UT']
for var in variables:
    label_encoder(data, var)

# Guarda el DataFrame normalizado
data.to_csv('data_select_normalizado.csv', index=False)

data = pd.read_csv("data_select_normalizado.csv")

#Guarda el DataFrame con valores faltantes en un archivo
data_con_faltantes = data[data.isnull().any(axis=1)]
data_con_faltantes.to_csv('data_select_con_faltantes.csv', index=False)

#Guarda el Dataframe sin valores faltantes en un archivo
data_sin_faltantes = data.dropna()
data_sin_faltantes.to_csv('data_select_sin_faltantes.csv', index=False)


