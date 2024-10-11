import pandas as pd

# Cargar los datos
data = pd.read_excel("Datos seleccionados.xlsx")

data.drop(['TOX_D_48_SUP_UT', 'TOX_D_48_FON_UT', 'TOX_FIS_SUP_15_UT',
          'TOX_FIS_FON_15_UT'], axis=1, inplace=True)

stadt = data.describe()

stadt.to_csv("Preprocesamiento\\Estadistica\\analisis_estadistico_SNI.csv")

