import pandas as pd

# Cargar los datos
data_lr_minmax = pd.read_csv("Preprocesamiento\\Union\\LR\\union_LR_MMS.csv")
data_lr_labelencoder = pd.read_csv("Preprocesamiento\\Union\\LR\\union_LR_LE.csv")

# Calcular estadísticas para los datos de imputación de LR con MinMax Scaling
stats_lr_minmax = data_lr_minmax.describe()

# Calcular estadísticas para los datos de imputación de LR con Label Encoder
stats_lr_labelencoder = data_lr_labelencoder.describe()

# Calcular correlaciones (Pearson y Spearman) para los datos de imputación de LR con MinMax Scaling
correlation_lr_minmax_pearson = data_lr_minmax.corr(method='pearson')
correlation_lr_minmax_spearman = data_lr_minmax.corr(method='spearman')

# Calcular correlaciones (Pearson y Spearman) para los datos de imputación de LR con Label Encoder
correlation_lr_labelencoder_pearson = data_lr_labelencoder.corr(method='pearson')
correlation_lr_labelencoder_spearman = data_lr_labelencoder.corr(method='spearman')

# Agregar etiquetas para diferenciar los resultados
stats_lr_minmax['Type'] = 'LR with MinMax Scaling'
stats_lr_labelencoder['Type'] = 'LR with Label Encoder'
correlation_lr_minmax_pearson['Type'] = 'LR with MinMax Scaling (Pearson)'
correlation_lr_minmax_spearman['Type'] = 'LR with MinMax Scaling (Spearman)'
correlation_lr_labelencoder_pearson['Type'] = 'LR with Label Encoder (Pearson)'
correlation_lr_labelencoder_spearman['Type'] = 'LR with Label Encoder (Spearman)'

# Concatenar todos los resultados en un solo DataFrame
all_results = pd.concat([stats_lr_minmax, stats_lr_labelencoder,
                         correlation_lr_minmax_pearson, correlation_lr_minmax_spearman,
                         correlation_lr_labelencoder_pearson, correlation_lr_labelencoder_spearman])

# Guardar el DataFrame en un archivo CSV
all_results.to_csv("Preprocesamiento\\Estadistica\\analisis_estadistico_LR.csv")
