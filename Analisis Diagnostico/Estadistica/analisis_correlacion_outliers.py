import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar los datos completos
data_complete = pd.read_csv("Analisis Diagnostico\\Estadistica\\data_complete.csv")
data_no_outliers = pd.read_csv("Analisis Diagnostico\\Estadistica\\data_no_outliers.csv")
outliers = pd.read_csv("Analisis Diagnostico\\Estadistica\\outlier.csv")

# Seleccionar solo las columnas numéricas
numeric_data = data_complete.select_dtypes(include=['float64', 'int64'])
numeric_data_no_outliers = data_no_outliers.select_dtypes(include=['float64', 'int64'])
numeric_data_only_outliers = outliers.select_dtypes(include=['float64', 'int64'])

# Calcular la correlación de Pearson
correlation_data_complete_pearson = numeric_data.corr(method= 'pearson')
correlation_data_no_outliers_pearson = numeric_data_no_outliers.corr(method='pearson')
correlation_data_only_outliers_pearson = numeric_data_only_outliers.corr(method='pearson')

print("Correlación de Pearson entre las variables (Datos Completos):")
print(correlation_data_complete_pearson)
print("\nCorrelación de Pearson entre las variables (Datos sin Outliers):")
print(correlation_data_no_outliers_pearson)
print("\nCorrelación de Pearson entre las variables (Solo Outliers):")
print(correlation_data_only_outliers_pearson)


# Visualizar la correlación gráficamente
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_data_complete_pearson, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlación de Pearson entre las variables (Datos Completos)")
plt.show()

# Visualizar la correlación de los no outliers gráficamente
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_data_no_outliers_pearson, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlación de Pearson entre las variables (Datos sin Outliers)")
plt.show()

# Visualizar la correlación de los outliers gráficamente
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_data_only_outliers_pearson, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlación de Pearson entre las variables (Solo Outliers)")
plt.show()
