import pandas as pd
# Cargar los datos
data = pd.read_excel("Datos seleccionados.xlsx")

# Eliminar las columnas no deseadas
data.drop(['TOX_D_48_SUP_UT', 'TOX_D_48_FON_UT', 'TOX_FIS_SUP_15_UT',
           'TOX_FIS_FON_15_UT'], axis=1, inplace=True)

# Función para detectar outliers usando IQR
def detect_outliers_iqr(df):
    outliers = pd.DataFrame(columns=df.columns)
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        outliers_in_col = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outliers = pd.concat([outliers, outliers_in_col], axis=0)
        
    return outliers

# Detectar outliers
outliers = detect_outliers_iqr(data)
outliers.to_csv("outliers.csv")

# Eliminar outliers
data_sin_outliers = data[~data.index.isin(outliers.index)]

# Análisis descriptivo del conjunto completo
print("Análisis descriptivo de los datos completos:")
print(data.describe())
data.describe().to_csv("Analisis Diagnostico\\Estadistica\\analisis_estadistico_datos_completos.csv")
data.to_csv("Analisis Diagnostico\\Estadistica\\datos_completos.csv")

# Análisis descriptivo de los datos sin outliers
print("\nAnálisis descriptivo de los datos sin outliers:")
print(data_sin_outliers.describe())
data_sin_outliers.describe().to_csv("Analisis Diagnostico\\Estadistica\\analisis_estadistico_datos_sin_outliers.csv")
data_sin_outliers.to_csv("Analisis Diagnostico\\Estadistica\\datos_sin_outliers.csv")
