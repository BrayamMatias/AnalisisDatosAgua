import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import LabelEncoder

# Caracteristicas: SITIO	DBO_mg/L	DQO_mg/L	SST_mg/L	COLI_FEC_NMP_100mL	E_COLI_NMP_100mL OD_PORC	TOX_D_48_UT	TOX_V_15_UT

# Carga el DataFrame con valores faltantes
data = pd.read_excel("Datos seleccionados.xlsx")
# Elimina las columnas que no se van a utilizar
data.drop(['TOX_D_48_SUP_UT', 'TOX_D_48_FON_UT', 'TOX_FIS_SUP_15_UT',
          'TOX_FIS_FON_15_UT'], axis=1, inplace=True)

# Lista de variables categóricas
variables_categoricas = ['SITIO']
# Lista de variables numéricas
variables_numericas = [col for col in data.columns if col not in variables_categoricas]

# Guarda las variables categóricas en un nuevo DataFrame
data_categorico = data[variables_categoricas]

# Aplica Label Encoding a las variables categóricas
for var in variables_categoricas:
    le = LabelEncoder()
    data_categorico.loc[:, var] = le.fit_transform(data_categorico[var])

# Normaliza las variables numéricas con Min-Max Scaling
scaler = MinMaxScaler()
data_numerico = scaler.fit_transform(data[variables_numericas])
data_numerico = pd.DataFrame(data_numerico, columns=variables_numericas)

# Combina las variables categóricas y numéricas
data_normalizado = pd.concat([data_categorico, data_numerico], axis=1)

# Guarda el DataFrame normalizado
data_normalizado.to_csv('data_select_normalizado_standar_scaling.csv', index=False)

# Guarda el DataFrame con valores faltantes en un archivo
data_con_faltantes = data_normalizado[data_normalizado.isnull().any(axis=1)]
data_con_faltantes.to_csv('data_select_con_faltantes_standar_scaling.csv', index=False)

# Guarda el DataFrame sin valores faltantes en un archivo
data_sin_faltantes = data_normalizado.dropna()
data_sin_faltantes.to_csv('data_select_sin_faltantes_standar_scaling.csv', index=False)
