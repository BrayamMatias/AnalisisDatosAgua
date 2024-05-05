#Cargar los archivos de los datos sin faltantes con los datos de la imputacion de datos

import pandas as pd
import numpy as np

#Cargar los datos sin faltantes
datos_sin_faltantes_LE = pd.read_csv('Preprocesamiento\\Normalización\\LabelEncoder\\data_select_sin_faltantes_LE.csv')
#Cargar los datos de la imputación
datos_imputados_KNN_LE = pd.read_csv('Preprocesamiento\\Imputación\\KNN\\data_select_imputed_KNN_LE.csv')

#Unir los datos despues de la ultima fila
datos_union_KNN_LE = pd.concat([datos_sin_faltantes_LE, datos_imputados_KNN_LE], axis=0)
#Guardar los datos
datos_union_KNN_LE.to_csv('Preprocesamiento\\Union\\KNN\\union_KNN_LE.csv', index=False)
