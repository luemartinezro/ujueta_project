### manipulate data

import pandas as pd

# carga de datas
datas = pd.read_csv(
    "/home/usuario/Escritorio/Consultorias_Empresariales/Ujueta/Datos/output_nov.csv"
)
# convertir en dataframe
df = pd.DataFrame(datas)
df.head()


# converti docdate to datetime
df["docdate"] = pd.to_datetime(df["docdate"])
#filter since 2022-01-01
df.docdate
dff2022 = df[df["docdate"] >= "2022-01-01"]

dff2022.shape
len(dff2022.notnull().sum())

# Count number of zeros in all columns of Dataframe
for col in dff2022.columns:
    column = dff2022[col]
    # Get the count of Zeros in column
    count = (column == 0).sum()
    print("Count of zeros in column ", col, " is : ", count)

# filter since 2023-01-01

dff2023 = df[df["docdate"] >= "2023-01-01"]

dff2023.shape
len(dff2023.notnull().sum())

# Count number of zeros in all columns of Dataframe
for col in dff2023.columns:
    column = dff2023[col]
    # Get the count of Zeros in column
    count = (column == 0).sum()
    print("Count of zeros in column ", col, " is : ", count)
    
#================================================================================================
# Crear grupos


data = dff2023.copy()
data.head()
# check null values
data.isnull().sum()

# review null values


values_list = ((data.isin([0]).sum(axis=1)) / len(data)) * 100
values_list

# created a separation with percentage of null values
data.apply( lambda s : s.value_counts().get(key=0,default=0), axis=1)
    
    
    
    

datahigh = []
datamedium = []
datalow = []

for col in data.columns:
    missing_percentage = (data[col].apply( lambda s : s.value_counts().get(key=0,default=0))) / len(data) * 100
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
