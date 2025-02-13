#%%
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, HuberRegressor, Lars, LassoLars, PassiveAggressiveRegressor, RANSACRegressor, SGDRegressor
from sklearn.model_selection import train_test_split

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

def detectar_atipicos(serie):
    """
    Detecta valores atípicos en una serie utilizando el rango intercuartílico (IQR).
    
    :param serie: Serie de pandas en la que se detectarán los valores atípicos.
    :return: Serie de pandas con 1 si el valor es atípico y 0 en caso contrario.
    """
    Q1 = serie.quantile(0.25)
    Q3 = serie.quantile(0.75)
    IQR = Q3 - Q1
    atipicos = ((serie < (Q1 - 1.5 * IQR)) | (serie > (Q3 + 1.5 * IQR))).astype(int)
    return atipicos

def calcular_variacion_porcentual(serie):
    """
    Calcula la variación porcentual de una serie.
    
    :param serie: Serie de pandas en la que se calculará la variación porcentual.
    :return: Serie de pandas con la variación porcentual.
    """
    variacion = serie.pct_change().fillna(0)
    return variacion

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

#%% Detectar valores atípicos y calcular variación porcentual por unique_id
df['atipico_y'] = df.groupby('unique_id')['y'].apply(detectar_atipicos).reset_index(level=0, drop=True)
df['atipico_stock'] = df.groupby('unique_id')['stock'].apply(detectar_atipicos).reset_index(level=0, drop=True)
df['atipico_margen'] = df.groupby('unique_id')['margen'].apply(detectar_atipicos).reset_index(level=0, drop=True)
#df['variacion_margen'] = df.groupby('unique_id')['margen'].apply(calcular_variacion_porcentual).reset_index(level=0, drop=True)

print(df.head())

# Extract month from ds and add as an exogenous variable
df['month'] = df['ds'].dt.month

#%%

# Define models to compare
models = {
    'LinearRegression': LinearRegression(),
    'Lasso': Lasso(),
    'Ridge': Ridge(),
    'ElasticNet': ElasticNet(),
    'HuberRegressor': HuberRegressor(),
    'Lars': Lars(),
    'LassoLars': LassoLars(),
    #'PassiveAggressiveRegressor': PassiveAggressiveRegressor(),
    #'RANSACRegressor': RANSACRegressor(),
    #'SGDRegressor': SGDRegressor(),
    #'XGBRegressor': XGBRegressor()
}



# Prepare data for modeling
X = df[['month', 'pandemia', 'atipico_y', 'atipico_stock', 'atipico_margen']]
y = df['y']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%
# Function to train and evaluate models
def train_evaluate_models(X_train, y_train, X_test, y_test):
    results = {}
    for name, model in models.items():
        pipeline = Pipeline([('model', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics = calcular_metricas(y_test, y_pred)
        results[name] = metrics
    return results, y_pred

#%%
# Train and evaluate models for each unique_id
best_models = {}
for unique_id in df['unique_id'].unique():
    df_unique = df[df['unique_id'] == unique_id]
    df_unique['month'] = df_unique['ds'].dt.month
    df_unique.fillna(0, inplace=True)
    X_unique = df_unique[['month', 'pandemia', 'atipico_y', 'atipico_stock', 'atipico_margen']]
    y_unique = df_unique['y']
    X_train, X_test, y_train, y_test = train_test_split(X_unique, y_unique, test_size=0.2, random_state=42)
    results, y_pred = train_evaluate_models(X_train, y_train, X_test, y_test)
    best_model = min(results, key=lambda k: results[k]['RMSE'])
    best_models[unique_id] = best_model

print(best_models)

#%%


# Create a table of exogenous variables for forecasting the next 6 months
last_date = df['ds'].max()
future_dates = pd.date_range(start=last_date, periods=7, freq='M')[1:]
future_df = pd.DataFrame({'ds': future_dates})

# Add exogenous variables to future_df
future_df['month'] = future_df['ds'].dt.month
future_df['pandemia'] = future_df['ds'].apply(lambda x: 1 if inicio_pandemia <= x <= fin_pandemia else 0)
# Assuming no outliers and no margin variation for future dates
future_df['atipico_y'] = 0
future_df['atipico_stock'] = 0
future_df['atipico_margen'] = 0
future_df['variacion_margen'] = 0

print(future_df)
# %%
