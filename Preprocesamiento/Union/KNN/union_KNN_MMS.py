#Cargar los archivos de los datos sin faltantes con los datos de la imputacion de datos

import pandas as pd
import numpy as np

#Cargar los datos sin faltantes
datos_sin_faltantes_MMS = pd.read_csv('Preprocesamiento\\Normalización\\MinMaxScaler\\data_select_sin_faltantes_MMS.csv')
#usar solo 4 decimales
datos_sin_faltantes_MMS = datos_sin_faltantes_MMS.round(4)
#Cargar los datos de la imputación
datos_imputados_KNN_MMS = pd.read_csv('Preprocesamiento\\Imputación\\KNN\\data_select_imputed_KNN_MMS.csv')

#Unir los datos despues de la ultima fila
datos_union_KNN_MMS = pd.concat([datos_sin_faltantes_MMS, datos_imputados_KNN_MMS], axis=0)
#Guardar los datos
datos_union_KNN_MMS.to_csv('Preprocesamiento\\Union\\KNN\\union_KNN_MMS.csv', index=False)
