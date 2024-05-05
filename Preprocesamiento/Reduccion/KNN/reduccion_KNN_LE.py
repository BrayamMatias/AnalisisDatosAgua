import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el conjunto de datos desde un archivo CSV
data = pd.read_csv('Preprocesamiento\\Union\\KNN\\union_KNN_LE.csv')

# Calcular la matriz de correlación
correlation_matrix = data.corr()

plt.figure(figsize=(10,10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# Definir umbral de correlación
umbral_correlacion = 0.6

# Encontrar las características que superan el umbral de correlación
high_corr_features = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if correlation_matrix.iloc[i, j] > umbral_correlacion:
            high_corr_features.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))

# Imprimir Características con alta correlación
print("Las características con correlación superior al umbral son:")
for feature_pair in high_corr_features:
    print(f"{feature_pair[0]} y {feature_pair[1]} con una correlación de {correlation_matrix.loc[feature_pair[0], feature_pair[1]]}")

# Guardar columnas con alta correlación en un archivo CSV
selected_columns = list(set([feature[0] for feature in high_corr_features] + [feature[1] for feature in high_corr_features]))
#selected_columns.append('Class') #Variable objetivo
selected_data = data[selected_columns]
selected_data.to_csv('Preprocesamiento\\Reduccion\\KNN\\reduccion_KNN_LE.csv', index=False)