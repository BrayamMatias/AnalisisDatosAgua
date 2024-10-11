import pandas as pd
import matplotlib.pyplot as plt

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

# Función para eliminar outliers usando IQR
def remove_outliers_iqr(df):
    df_clean = df.copy()
    for col in df_clean.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

# Visualización de los datos completos con scatter plot
def visualize_all_data(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Graficar un scatter plot por cada columna numérica
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(df[col])), df[col], alpha=0.5)
        plt.title(f"Valores de la columna {col} por número de muestra (Datos completos)")
        plt.xlabel("Número de muestra")
        plt.ylabel(col)
        plt.show()

# Visualización de los outliers solamente
def visualize_outliers(df, outliers):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        plt.scatter(outliers.index, outliers[col], color='red', alpha=0.5)
        plt.title(f"Valores de la columna {col} por número de muestra (Outliers)")
        plt.xlabel("Número de muestra")
        plt.ylabel(col)
        plt.show()

# Visualización de los datos sin outliers
def visualize_data_without_outliers(df_clean):
    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
    
    # Graficar un scatter plot por cada columna numérica
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(df_clean[col])), df_clean[col], alpha=0.5)
        plt.title(f"Valores de la columna {col} por número de muestra (Sin outliers)")
        plt.xlabel("Número de muestra")
        plt.ylabel(col)
        plt.show()

# Detectar outliers
outliers = detect_outliers_iqr(data)
print("Valores atípicos detectados\n:", outliers)

# Guardar los outliers en un archivo CSV
outliers.to_csv("Preprocesamiento\\outlier.csv")

# Remover los outliers del conjunto de datos
data_sin_outliers = remove_outliers_iqr(data)
data_sin_outliers.to_csv("Analisis Diagnostico\\Estadistica\\data_no_outliers.csv")

# Visualizar los datos completos
visualize_all_data(data)

# Visualizar solo los outliers
visualize_outliers(data, outliers)

# Visualizar los datos sin outliers
visualize_data_without_outliers(data_sin_outliers)
