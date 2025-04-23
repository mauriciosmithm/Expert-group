# Databricks notebook source
from functools import reduce  # Manejo de listas
from pyspark.sql.functions import unix_timestamp  # Formatos de fecha
from pyspark.sql.functions import col #Manipulación de columnas
from pyspark.sql.functions import when, lit #Condiciones y remplazos en Data Frames
from pyspark.sql.functions import regexp_replace # Remplazar Expresiones regulares
from pyspark.sql.functions import upper # Upper
from pyspark.sql.functions import trim # Trim
from pyspark.sql.functions import unix_timestamp, from_unixtime

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/clothes_price_prediction_dat.csv"
file_type = "csv"

df_raw = spark.read.format("csv").options(header=True, delimiter=';', encoding='utf-8').load(file_location)



# Filtrar columnas que **NO** comienzan con '_c'
df_cleaned = df_raw.select([col(c) for c in df_raw.columns if not c.startswith("_c")])


# Obtener la primera fila como encabezado
first_row = df_cleaned.first()
columns = [str(cell).strip() for cell in first_row]

# Aplicar los nuevos nombres de columna
df_renamed = df_cleaned.rdd.zipWithIndex().filter(lambda row: row[1] > 0).keys().toDF(columns)


display(df_renamed)

# COMMAND ----------

# MAGIC %md
# MAGIC ## o	Realizar limpieza de datos (eliminación de valores nulos y duplicados).

# COMMAND ----------

from pyspark.sql.functions import split, col

# Nombre original de la columna (con coma)
col_name = "Brand,Category,Color,Size,Material,Price"

# Separar los valores en una nueva columna
df_split = df_renamed.withColumn("split_col", split(col(f"`{col_name}`"), ","))

# Asignar los elementos a nuevas columnas
df_final = df_split.select(
    col("split_col")[0].alias("Brand"),
    col("split_col")[1].alias("Category"),
    col("split_col")[2].alias("Color"),
    col("split_col")[3].alias("Size"),
    col("split_col")[4].alias("Material"),
    col("split_col")[5].alias("Price")
)

df_final.show(truncate=False)

# COMMAND ----------

from pyspark.sql.types import IntegerType

# Convertir la columna Price a entero
df_final = df_final.withColumn("Price", col("Price").cast(IntegerType()))

df_final.printSchema()
df_final.show(truncate=False)

# COMMAND ----------

df_final_sin_duplicados = df_final.dropDuplicates()

# COMMAND ----------

# MAGIC %md
# MAGIC ## o	Exploración inicial de los datos para identificar patrones y relaciones clave.

# COMMAND ----------

#Cruce de columnas para ver patrones

df_final_sin_duplicados.groupBy("Color", "Material").count().orderBy("count", ascending=False).show()

# COMMAND ----------

#Estadísticas descriptivas para la columna Price
df_final_sin_duplicados.describe("Price").show()

# COMMAND ----------

from pyspark.sql.functions import skewness, kurtosis

df_final_sin_duplicados.select(
    skewness("Price").alias("Skewness"),
    kurtosis("Price").alias("Kurtosis")
).show()

# COMMAND ----------

for col_name in df_final_sin_duplicados.columns:
    print(f"{col_name}: {df_final_sin_duplicados.select(col_name).distinct().count()} valores únicos")

# COMMAND ----------

df_final_sin_duplicados.groupBy("Brand").count().orderBy("count", ascending=False).show()
df_final_sin_duplicados.groupBy("Category").count().orderBy("count", ascending=False).show()

# COMMAND ----------

#CONEVRTIR EL DATAFRAME EN UN ARCHIVO PARQUET
df_final_sin_duplicados.write.format('parquet').mode('overwrite').save('/FileStore/tables/archivos')

# COMMAND ----------

df_final_sin_duplicados.repartition(1).write.format('parquet').mode('overwrite').save('/FileStore/tables/resultados')

# COMMAND ----------

display(dbutils.fs.ls("/FileStore/tables/resultados"))

# COMMAND ----------

df=spark.read.parquet("/FileStore/tables/archivos")
display(df)

# COMMAND ----------

