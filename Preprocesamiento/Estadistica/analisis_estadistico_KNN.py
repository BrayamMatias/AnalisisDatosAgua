import pandas as pd

# Cargar los datos
data_knn_minmax = pd.read_csv("Preprocesamiento\Imputación\KNN\data_select_imputed_KNN_MMS.csv")
data_knn_labelencoder = pd.read_csv("Preprocesamiento\Imputación\KNN\data_select_imputed_KNN_LE.csv")

# Calcular estadísticas para los datos de imputación de KNN con MinMax Scaling
stats_knn_minmax = data_knn_minmax.describe()

# Calcular estadísticas para los datos de imputación de KNN con Label Encoder
stats_knn_labelencoder = data_knn_labelencoder.describe()

# Calcular correlaciones (Pearson y Spearman) para los datos de imputación de KNN con MinMax Scaling
correlation_knn_minmax_pearson = data_knn_minmax.corr(method='pearson')
correlation_knn_minmax_spearman = data_knn_minmax.corr(method='spearman')

# Calcular correlaciones (Pearson y Spearman) para los datos de imputación de KNN con Label Encoder
correlation_knn_labelencoder_pearson = data_knn_labelencoder.corr(method='pearson')
correlation_knn_labelencoder_spearman = data_knn_labelencoder.corr(method='spearman')

# Agregar etiquetas para diferenciar los resultados
stats_knn_minmax['Type'] = 'KNN with MinMax Scaling'
stats_knn_labelencoder['Type'] = 'KNN with Label Encoder'
correlation_knn_minmax_pearson['Type'] = 'KNN with MinMax Scaling (Pearson)'
correlation_knn_minmax_spearman['Type'] = 'KNN with MinMax Scaling (Spearman)'
correlation_knn_labelencoder_pearson['Type'] = 'KNN with Label Encoder (Pearson)'
correlation_knn_labelencoder_spearman['Type'] = 'KNN with Label Encoder (Spearman)'

# Concatenar todos los resultados en un solo DataFrame
all_results = pd.concat([stats_knn_minmax, stats_knn_labelencoder,
                         correlation_knn_minmax_pearson, correlation_knn_minmax_spearman,
                         correlation_knn_labelencoder_pearson, correlation_knn_labelencoder_spearman])

# Guardar el DataFrame en un archivo CSV
all_results.to_csv("all_results.csv")
