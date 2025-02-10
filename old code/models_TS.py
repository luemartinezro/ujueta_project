### manipulate data
try:
    import pandas as pd
    import numpy as np

    # from pyramid.arima import auto_arima
    from statsmodels.tsa.arima_model import ARIMA
    import pmdarima as pm
    from pmdarima.arima import auto_arima
    from sklearn.metrics import mean_squared_error


except Exception as e:
    print("Library not found: ", e)


# carga de datas
data = pd.read_csv(
    "/home/usuario/Documentos/Consultorias_Empresariales/Ujueta/Datos/output.csv"
)
# convertir en dataframe

df = pd.DataFrame(data)

# =====================================================================================
# revisiones generales
df.shape
print(df.duplicated().sum())
print(df.groupby(["docdate", "codigoarticulo"]).size())

# check unique valus
df.value_counts()

# Unique to define values
df["dbpais"].unique()


# ===========================================================================================
### Casteos de variables - filtro con grupo-ujueta
# converti docdate to datetime
df["docdate"] = pd.to_datetime(df["docdate"])
# garantizar que quantity sea numerico
df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

# create a dataframe only for colombia
df = df[
    (df["dbpais"] == "GRUPO_UJUETA")
]  # & (df["dbpais"]=='VENTAS_INT') & (df["dbpais"]=='UJUETA_TRADING')]
df

### Pivot table>>> crear la tabla para los pronosticos
# pivot el dataframe
pivot_df = df.pivot_table(
    index="docdate",
    columns="codigoarticulo",
    values="quantity",
    aggfunc="first",
    fill_value=0,
)  # aggfunc='sum'

# remplazar los valores nulos con 0 (o con otro valor que se apropiado)
pivot_df = pivot_df.fillna(0)

# check data
pivot_df

pivot_df.describe()

# check no null values per year
dff = pivot_df.reset_index(level="docdate", col_level=1)

dff = pd.DataFrame(dff)
dff.info()
dff.isnull().sum()

# filter since 2022-01-01
dff.docdate
dff2022 = dff[dff["docdate"] >= "2023-01-01"]

dff2022.shape
len(dff2022.notnull().sum())
dff2022.isna().sum()


# ========================================================================================================
# ========================================================================================================
### skforecast exercise

# https://www.kaggle.com/code/swapnilagnihotri/multi-series-forecasting-using-skforecast

try:
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf

    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingRegressor

    from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
    from skforecast.ForecasterAutoreg import ForecasterAutoreg
    from skforecast.model_selection import backtesting_forecaster
    from skforecast.model_selection import grid_search_forecaster
    from skforecast.model_selection_multiseries import (
        backtesting_forecaster_multiseries,
    )
    from skforecast.model_selection_multiseries import (
        grid_search_forecaster_multiseries,
    )
except Exception as e:
    print("Library not found: ", e)


# carga de datas
data = pd.read_csv(
    "/home/usuario/Documentos/Consultorias_Empresariales/Ujueta/Datos/output.csv"
)
# convertir en dataframe
df = pd.DataFrame(data)

# =====================================================================================
# revisiones generales
df.shape
print(df.duplicated().sum())
print(df.groupby(["docdate", "codigoarticulo"]).size())

# check unique valus
df.value_counts()

# Unique to define values
df["dbpais"].unique()


# ===========================================================================================
### Casteos de variables - filtro con grupo-ujueta
# converti docdate to datetime
df["docdate"] = pd.to_datetime(df["docdate"])
# garantizar que quantity sea numerico
# df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
df["quantity"] = df["quantity"].astype(int)

# create a dataframe only for colombia
df = df[
    (df["dbpais"] == "GRUPO_UJUETA")
]  # & (df["dbpais"]=='VENTAS_INT') & (df["dbpais"]=='UJUETA_TRADING')]
df
df.info()

# Data preprocessing

df["docdate"] = pd.to_datetime(df["docdate"], format="%Y-%m-%d")


# pivot el dataframe
# data = df.pivot_table( index="docdate", columns="codigoarticulo",
#                    values="quantity", aggfunc="first",fill_value=0,)  # aggfunc='sum'


data = df.pivot_table(values="quantity", index="docdate", columns="codigoarticulo")
data.columns.name = None
data.columns = [col for col in data.columns]
data = data.asfreq("1D")
data = data.sort_index()
data.head(5)

# check no null values per year
data = data.reset_index(level="docdate", col_level=1)

data = data[data["docdate"] >= "2023-01-01"]
data.describe()
data.info()

# check null values
data.isnull().sum()

# review null values

values_list = ((data.isnull().sum()) / len(data)) * 100
values_list

# created a separation with percentage of null values
datahigh = []
datamedium = []
datalow = []

for col in data.columns:
    missing_percentage = data[col].isnull().sum() / len(data) * 100
    if missing_percentage < 30.0:
        datahigh.append(col)
    elif missing_percentage < 50.0:
        datamedium.append(col)
    else:
        datalow.append(col)

# check selections of dataset
datahigh
datamedium
datalow

len(datahigh)
len(datamedium)
len(datalow)

# Created a dataset with high, medium, low quantity of true values
datas_h = data[datahigh]
datas_h.describe()
datas_h.shape

datas_m = data[datamedium]
datas_m.describe()
datas_m.shape

datas_l = data[datalow]
datas_l.describe()
datas_l.shape


# dropna with a thresh
# data = data.dropna(axis=1, thresh=400)

# datas_h = datas_h.bfill(axis=0)

datas_h = datas_h.interpolate(method="linear", limit_direction="backward")
# imputation values

# check null values
datas_h.isnull().sum()

datas_h.isna().sum()


# ===============================
# sma


def sma_predictions(data, window, train_len):

    y_hat_sma = data.copy()

    for col in data.columns:
        y_hat_sma[f"sma_forecast_{col}"] = data[col].rolling(window).mean()
        y_hat_sma[f"sma_forecast_{col}"][train_len:] = y_hat_sma[f"sma_forecast_{col}"][
            train_len - 1
        ]

    return y_hat_sma


def sma_predictions(data, window, train_len):
    # Compute the rolling mean for the entire DataFrame
    rolling_mean = data.rolling(window).mean()

    # Create a copy to store predictions
    y_hat_sma = data.copy()

    # Create the forecast columns
    forecast_columns = {
        f"sma_forecast_{col}": rolling_mean[col] for col in data.columns
    }

    # Assign rolling means to the new columns
    for col, forecast in forecast_columns.items():
        y_hat_sma[col] = forecast
        # Set the values after the training length to the last value of the training set
        y_hat_sma.loc[train_len:, col] = forecast.iloc[train_len - 1]

    return y_hat_sma


y_hat = sma_predictions(datas_h.iloc[:, 2:], 30, 537)

y_hat


# =======================================================================================================
# Arima model

cols = datas_h.columns
model = []

for i in cols:
    model.append(
        tuple(
            (
                pm.auto_arima(
                    datas_h.loc[:, i],
                    start_p=1,
                    start_q=1,
                    test="adf",  # Use el test adftest para hallar el óptimo de "d"
                    max_p=5,
                    max_q=5,  # maximos de p y q
                    m=365,  # frecuencia de la serie
                    d=None,  # Dejar que el modelo determine el "d"
                    seasonal=False,  # No estacional
                    start_P=0,
                    D=0,
                    trace=True,
                    error_action="ignore",
                    suppress_warnings=True,
                    stepwise=True,
                    scoring="mse",
                    scoring_args=None,
                ),
                i,
            )
        )
    )
model.summary()

n = len(data.index)

prediction = pd.DataFrame(model[0][0].predict(n_periods=n), index=data.index)
prediction = pd.DataFrame(
    model[0][0].predict(n_periods=n),
    index=pd.date_range(start="2022-06-20", periods=n, freq="D"),
)

N = len(model) / 2
N = len(model)
d = np.linspace(0, N, N, endpoint=True, dtype=int)

for j in d:
    prediction[cols[j]] = model[j][0].predict(n_periods=n)


prediction


# data = data.drop(columns=['ACEROCOMBO001', 'ADDW44830', 'ADDW44840'])

# split data into train-val-test
import math

train_len = math.ceil(len(data) * 0.80)
train = data[0:train_len]
test = data[train_len:]


print(f"Train dates      : {train.index.min()}---{train.index.max()} (n={len(train)})")
print(f"Test dates       : {test.index.min()} ---{test.index.max()} (n={len(test)})")


# ####

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Assuming your data is in a DataFrame with columns 'time_series_id', 'date', and 'value'
# data = data.pivot(index='date', columns='time_series_id', values='value')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(datas_h)


# Create sequences
def create_sequences(data, time_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : (i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)


time_steps = 10
X, y = create_sequences(scaled_data, time_steps)

# Reshape input to be [samples, time steps, features]
X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

# Define LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_steps, X.shape[2])))
model.add(LSTM(50))
model.add(Dense(X.shape[2]))
model.compile(optimizer="adam", loss="mse")

# Fit the model
model.fit(X, y, epochs=20, batch_size=32, verbose=1)

# Forecast
forecast = model.predict(X[-1].reshape(1, time_steps, X.shape[2]))
forecast = scaler.inverse_transform(forecast)
forecast


# Train and backtest a model for all items: ForecasterAutoregMultiSeries
# ========================================================================
#

# items = list(datas_h.columns)
#
### Define forecaster
# forecaster_ms = ForecasterAutoregMultiSeries(
#                regressor          = HistGradientBoostingRegressor(random_state=123),
#                lags               = 15,
#                transformer_series = StandardScaler(),
#                )
#
## Backtesting forecaster for all items
# multi_series_mae, predictions_ms = backtesting_forecaster_multiseries(
#                                        forecaster           = forecaster_ms,
#                                        series               = datas_h,
#                                        levels               = items,
#                                        steps                = 7,
#                                        metric               = 'mean_absolute_error',
#                                        initial_train_size   = len(train),
#                                        refit                = False,
#                                        fixed_train_size     = False,
#                                        verbose              = False
#
#                                    )
#
### Results
# print(multi_series_mae.head())
# print("")
# print(predictions_ms.head())
