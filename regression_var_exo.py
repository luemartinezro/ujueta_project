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

def detectar_atipicos(df, columna):
    """
    Detecta valores atípicos en una columna de un DataFrame utilizando el rango intercuartílico (IQR).
    
    :param df: DataFrame de pandas.
    :param columna: Nombre de la columna en la que se detectarán los valores atípicos.
    :return: Serie de pandas con 1 si el valor es atípico y 0 en caso contrario.
    """
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1
    atipicos = ((df[columna] < (Q1 - 1.5 * IQR)) | (df[columna] > (Q3 + 1.5 * IQR))).astype(int)
    return atipicos

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

#%% Detectar valores atípicos por unique_id
df['atipico_y'] = df.groupby('unique_id')['y'].transform(detectar_atipicos, columna='y')
df['atipico_stock'] = df.groupby('unique_id')['stock'].transform(detectar_atipicos, columna='stock')
df['atipico_margen'] = df.groupby('unique_id')['margen'].transform(detectar_atipicos, columna='margen')

print(df.head())

#%% Calcular la variación porcentual de la variable margen por cada unique_id
df['variacion_margen'] = df.groupby('unique_id')['margen'].transform(lambda x: x.pct_change())
df['variacion_margen'] = df['variacion_margen'].fillna(0)
print(df.head())