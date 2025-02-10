#%%

# Libraries
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor

from skforecast.datasets import fetch_dataset
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.model_selection import TimeSeriesFold
from skforecast.model_selection import backtesting_forecaster_multiseries
from skforecast.model_selection import grid_search_forecaster_multiseries
from skforecast.model_selection import bayesian_search_forecaster_multiseries

#%%
#-- Funciones

def leer_archivo_excel(ruta_archivo, hoja=None):
    """
    Lee un archivo Excel y devuelve un DataFrame de pandas.
    
    :param ruta_archivo: Ruta del archivo Excel.
    :param hoja: Nombre o índice de la hoja a leer. Si es None, se lee la primera hoja.
    :return: DataFrame con los datos del archivo Excel.
    """
    df = pd.read_excel(ruta_archivo, sheet_name=hoja)
    return df

#-- Lectura de datos 
ruta_archivo = '../Ejercicio Forecast.xlsx'
df = leer_archivo_excel(ruta_archivo, hoja='Forecast')

print(df.info())
print(df.head())

#%% 
#-- Ajuste de nombres de variables
df.rename(columns={'fecha': 'ds', 'codigoarticulo': 'unique_id', 'cantidad': 'y'}, inplace=True)

# Crear una nueva variable que contemple los meses de pandemia por COVID-19
# Definir el rango de fechas de la pandemia (ajusta según sea necesario)
inicio_pandemia = pd.to_datetime('2020-03-01')
fin_pandemia = pd.to_datetime('2021-06-30')

# Crear la columna 'pandemia' que indica si la fecha está dentro del rango de la pandemia
df['pandemia'] = df['ds'].apply(lambda x: 1 if inicio_pandemia <= x <= fin_pandemia else 0)

print(df.head())

#%% Pivotar el DataFrame para tener los datos de codigoarticulo por columna
df_pivot = df.pivot(index=['fecha', 'stock', 'margen'], columns='codigoarticulo', values='cantidad').reset_index()
df_pivot.fillna(0, inplace=True)
print(df_pivot.head())
# %%
