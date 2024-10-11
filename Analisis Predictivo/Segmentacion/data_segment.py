import pandas as pd

# Carga el DataFrame con valores faltantes
data = pd.read_csv("Analisis Diagnostico\\Estadistica\\datos_completos.csv")

#Selecciona solo datos numericos
data_numeric = data.select_dtypes(include=['float64', 'int64'])

#Separa los datos con valores faltantes
data_con_faltantes = data[data_numeric.isnull().any(axis=1)]
data_con_faltantes.to_csv('Analisis Predictivo\\Segmentacion\\data_con_faltantes.csv', index=False)

#Separa los datos sin valores faltantes
data_sin_faltantes = data.dropna(subset=data_numeric.columns)
data_sin_faltantes.to_csv('Analisis Predictivo\\Segmentacion\\data_sin_faltantes.csv', index=False)
