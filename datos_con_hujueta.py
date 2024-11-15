"""
base de postgressql:
Host: 144.24.14.252: port 56000
Database: postgres
Authentification: Database Native
Nombre de usuario: consultoria
Contraseña: LmKTXJBXya!14]f9!2k]    

Host:pg.ujueta.com
Port: 5432
Database: postgres
Authentification: Database Native
Nombre de usuario: consultoria
Contraseña: LmKTXJBXya!14]f9!2k]



"""

import psycopg2
import pandas as pd
import csv

### verify the conexion
try:
    # Connect to the database
    conn = psycopg2.connect(
        host="pg.ujueta.com",
        port="5432",
        user="consultoria",
        password="LmKTXJBXya!14]f9!2k]",
        database="postgres",
    )
except psycopg2.Error as e:
    print("Error connecting to the database:")
    print(e)
else:
    print("Connection established successfully")



###
def export_forecast_to_csv():
    try:
        # connect to the database
        conn = psycopg2.connect(
            host="pg.ujueta.com",
            port="5432",
            user="consultoria",
            password="LmKTXJBXya!14]f9!2k]",
            database="postgres",
        )

        # create cursor
        with conn.cursor() as cur:

            # ececute a sql query

            cur.execute("SELECT * FROM forecast.ventas_diario")

            # fetch the results
            results = cur.fetchall()

            # open a file in the downloads folder

            with open(
                "/home/usuario/Escritorio/Consultorias_Empresariales/Ujueta/Datos/output.csv",
                "w",
                newline="",
            ) as f:
                # create a csv writer
                writer = csv.writer(f)

                # write the column names
                writer.writerow([col[0] for col in cur.description])

                # write the query results
                writer.writerows(results)
    except Exception as e:
        print(f"An error ocurred: {e}")
    finally:
        # close the cursor and connection
        if conn:
            conn.close()


# Call the function to export data to csv
export_forecast_to_csv()


### manipulate data

import pandas as pd

# carga de datas
data = pd.read_csv(
    "/home/usuario/Escritorio/Consultorias_Empresariales/Ujueta/Datos/output.csv"
)
# convertir en dataframe
df = pd.DataFrame(data)

# review data
df.shape
print(df.duplicated().sum())
print(df.groupby(["docdate", "codigoarticulo"]).size())

# check unique valus
df.value_counts()

# Unique to define values
df["dbpais"].unique()


# converti docdate to datetime
df["docdate"] = pd.to_datetime(df["docdate"])
# garantizar que quantity sea numerico
df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

# create a dataframe only for colombia
df = df[
    (df["dbpais"] == "GRUPO_UJUETA")
]  # & (df["dbpais"]=='VENTAS_INT') & (df["dbpais"]=='UJUETA_TRADING')]
df
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

# complete dataset: start 2021-01-01
dff.shape
len(dff.notnull().sum())

# Count number of zeros in all columns of Dataframe
for col in dff.columns:
    column = dff[col]
    # Get the count of Zeros in column
    count = (column == 0).sum()
    print("Count of zeros in column ", col, " is : ", count)

# filter since 2022-01-01
dff.docdate
dff2022 = dff[dff["docdate"] >= "2022-01-01"]

dff2022.shape
len(dff2022.notnull().sum())

# Count number of zeros in all columns of Dataframe
for col in dff2022.columns:
    column = dff2022[col]
    # Get the count of Zeros in column
    count = (column == 0).sum()
    print("Count of zeros in column ", col, " is : ", count)

# filter since 2023-01-01

dff2023 = dff[dff["docdate"] >= "2023-01-01"]

dff2023.shape
len(dff2023.notnull().sum())

# Count number of zeros in all columns of Dataframe
for col in dff2023.columns:
    column = dff2023[col]
    # Get the count of Zeros in column
    count = (column == 0).sum()
    print("Count of zeros in column ", col, " is : ", count)


