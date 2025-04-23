# Databricks notebook source
df=spark.read.parquet("/FileStore/tables/archivos")
display(df)

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

# 1. Crear sesión Spark (ya viene por defecto en Databricks, se incluye por claridad)
#spark = SparkSession.builder.getOrCreate()

# 2. Leer archivo Parquet
df = spark.read.parquet("/FileStore/tables/archivos")

# 3. Mostrar datos
display(df)
df.printSchema()

# 4. Convertir columna Price a entero si es necesario
from pyspark.sql.functions import col
df = df.withColumn("Price", col("Price").cast("int"))

# 5. Eliminar nulos (opcional según tus datos)
df = df.na.drop(subset=["Brand", "Category", "Color", "Size", "Material", "Price"])

# 6. Definir columnas categóricas
categorical_cols = ["Brand", "Category", "Color", "Size", "Material"]

# 7. Indexar y codificar variables categóricas
indexers = [StringIndexer(inputCol=c, outputCol=c + "_Index", handleInvalid="keep") for c in categorical_cols]
encoders = [OneHotEncoder(inputCol=c + "_Index", outputCol=c + "_Vec") for c in categorical_cols]

# 8. Ensamblar características
feature_cols = [c + "_Vec" for c in categorical_cols]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# 9. Modelo de regresión
lr = LinearRegression(featuresCol="features", labelCol="Price")

# 10. Crear pipeline
pipeline = Pipeline(stages=indexers + encoders + [assembler, lr])

# 11. Dividir datos
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# 12. Entrenar modelo
model = pipeline.fit(train_data)

# 13. Predecir sobre el conjunto de prueba
predictions = model.transform(test_data)

# 14. Evaluar modelo
evaluator = RegressionEvaluator(labelCol="Price", predictionCol="prediction")

rmse = evaluator.setMetricName("rmse").evaluate(predictions)
r2 = evaluator.setMetricName("r2").evaluate(predictions)

print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# 15. Mostrar algunas predicciones
display(predictions.select("Brand", "Category", "Color", "Size", "Material", "Price", "prediction"))


# COMMAND ----------

# MAGIC %md
# MAGIC el modelo de machine learning ha estimado que el precio de este producto, teniendo en cuenta su marca, categoría, color, talla y material, será de aproximadamente $####. Esta predicción se basa en patrones históricos aprendidos a partir de datos similares.

# COMMAND ----------

# Guardar modelo en DBFS
model.write().overwrite().save("/FileStore/modelos/regresion_precio_productos")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resultados del modelo 

# COMMAND ----------

from pyspark.ml import PipelineModel

# Cargar modelo desde DBFS
model_loaded = PipelineModel.load("/FileStore/modelos/regresion_precio_productos")

# Usar el modelo para predecir
predictions_loaded = model_loaded.transform(test_data)
display(predictions_loaded.select("Brand", "Category", "Color", "Size", "Material", "Price", "prediction"))

# COMMAND ----------

display(dbutils.fs.ls("/FileStore/modelos/regresion_precio_productos"))

# COMMAND ----------

from pyspark.ml import PipelineModel

# Cargar el modelo desde DBFS
model_loaded = PipelineModel.load("/FileStore/modelos/regresion_precio_productos")

# Realizar predicciones y guardar el resultado en un DataFrame
predictions_df = model_loaded.transform(test_data).select(
    "Brand", "Category", "Color", "Size", "Material", "Price", "prediction"
)

# Mostrar resultado
display(predictions_df)

# COMMAND ----------

