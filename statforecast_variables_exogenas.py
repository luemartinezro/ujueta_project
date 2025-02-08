#%%
#-- Paquetes

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error

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

# Convertir la columna de fechas para que muestre el primer día del mes
df['ds'] = pd.to_datetime(df['ds']).dt.to_period('M').dt.to_timestamp()

# Crear una nueva variable que contemple los meses de pandemia por COVID-19
# Definir el rango de fechas de la pandemia (ajusta según sea necesario)
inicio_pandemia = pd.to_datetime('2020-03-01')
fin_pandemia = pd.to_datetime('2021-06-30')

# Crear la columna 'pandemia' que indica si la fecha está dentro del rango de la pandemia
df['pandemia'] = df['ds'].apply(lambda x: 1 if inicio_pandemia <= x <= fin_pandemia else 0)

print(df.head())

#%% Filtrar codigoarticulo con la misma cantidad de filas
conteo_filas = df.groupby('unique_id').size().reset_index(name='conteo')
max_filas = conteo_filas['conteo'].max()
print(conteo_filas)
#df_filtrado = df[df['unique_id'].isin(conteo_filas[conteo_filas['conteo'] == max_filas]['unique_id'])]

df_filtrado = df[df['unique_id'].isin(['AUACSH1000', 'HEELAG1141', 
'MAEL2G65', 'SOELCSVM510', 'SOFUFW181'])]

print(df_filtrado.head())

#%% Filtrado de Items
# Preparar los datos para el modelo
df_filtrado = df_filtrado.sort_values(by=['unique_id', 'ds'])
df_filtrado = df_filtrado.fillna(0)


#%% Entrenamiento del modelo con StatsForecast
# Crear una instancia del modelo AutoARIMA

# Extract dates for train and test set 
dates = df_filtrado['ds'].unique()
dtrain = dates[:-6]
dtest = dates[-6:]

train = df_filtrado.query('ds in @dtrain')
X_test = df_filtrado.query('ds in @dtest') 
X_test.drop('y', axis=1, inplace=True, errors='ignore')
y_test = df_filtrado.query('ds in @dtest')[['ds','unique_id', 'y']].copy()

models = [AutoARIMA(season_length = 12)]
sf = StatsForecast(
    models=models, 
    freq='MS', 
    n_jobs=-1
)

fcst = sf.forecast(df=train, h=6, X_df=X_test, level=[95])
fcst = fcst.reset_index()
fcst.head()


# %%
StatsForecast.plot(df_filtrado, fcst, max_insample_length=6*2)

# %% MAE 
res = y_test.merge(fcst, how='left', on=['unique_id', 'ds'])
mae = abs(res['y']-res['AutoARIMA']).mean()
print('The MAE with exogenous regressors is '+str(round(mae,2)))

#-- Calcular el MAPE

mape = mean_absolute_percentage_error(res['y'], res['AutoARIMA'])
print(f'MAPE: {mape:.2%}')

# %%
