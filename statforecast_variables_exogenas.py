#%%
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error

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

def calcular_metricas(y_true, y_pred):
    """
    Calcula RMSE, MAE, MAPE y SMAPE.
    
    :param y_true: Valores reales.
    :param y_pred: Valores predichos.
    :return: Diccionario con las métricas.
    """
    rmse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    smape = 100 * (2 * abs(y_true - y_pred) / (abs(y_true) + abs(y_pred))).mean()
    
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'SMAPE': smape}

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
df_filtrado = df[df['unique_id'].isin(conteo_filas[conteo_filas['conteo'] == max_filas]['unique_id'])]

print(df_filtrado.head())

#%% Partición en train y test
# Definir el tamaño del conjunto de prueba
test_size = 6  # Ajusta según sea necesario

# Crear conjuntos de entrenamiento y prueba
train = df_filtrado.groupby('unique_id').apply(lambda x: x.iloc[:-test_size]).reset_index(drop=True)
test = df_filtrado.groupby('unique_id').apply(lambda x: x.iloc[-test_size:]).reset_index(drop=True)

print(train.head())
print(test.head())

#%% Entrenamiento del modelo con StatsForecast
# Preparar los datos para el modelo
train = train.sort_values(by=['unique_id', 'ds'])
train = train.fillna(0)

X_test = test.drop(columns='y').sort_values(by=['unique_id', 'ds'])
# Crear una instancia del modelo AutoARIMA
models = [AutoARIMA(season_length=12)]  # Ajusta el parámetro season_length según sea necesario

# Crear una instancia de StatsForecast
sf = StatsForecast(
    models=models, 
    freq='M', 
    n_jobs=1,
)


#-- Entrenar el modelo
horizon = test_size
level = [95]

fcst = sf.forecast(df=train, h=horizon, X_df=X_test, level=level)
fcst.reset_index(drop=False, inplace=True)
fcst.head()

#%% Calcular las métricas por unique_id
resultados = []

for unique_id in test['unique_id'].unique():
    y_true = test[test['unique_id'] == unique_id]['y']
    y_pred = fcst[fcst['unique_id'] == unique_id]['AutoARIMA']
    
    metricas = calcular_metricas(y_true, y_pred)
    metricas['unique_id'] = unique_id
    resultados.append(metricas)

df_metricas = pd.DataFrame(resultados)
print(df_metricas)

#%% Crear tabla exógena para los pronósticos futuros

