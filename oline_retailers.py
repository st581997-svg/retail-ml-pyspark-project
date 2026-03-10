from pyspark.sql import SparkSession

spark = SparkSession.builder\
   .appName("RetailDataAnalysis")\
   .getOrCreate()

df = spark.read.csv("Online Retail.csv",header=True,
                    inferSchema=True)

df.show(10)
df.printSchema()


from pyspark.sql.functions import col

# Remove missing CustomerID
df_clean = df.dropna(subset=["CustomerID"])

# Remove duplicates
df_clean = df_clean.dropDuplicates()

# Remove negative quantity
df_clean = df_clean.filter(col("Quantity") > 0)

# Remove zero price
df_clean = df_clean.filter(col("UnitPrice") > 0)

df_clean.show()

# feature engineering 

#TOTAL PURCHASE VALUE 

from pyspark.sql.functions import col

df_clean = df_clean.withColumn(
    "TotalAmount",
    col("Quantity") *col("UnitPrice")
)

# LABEL & FEATURE SELECTION 

# predict total amount using quantity and unitprice 
import numpy as np
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler( 
    inputCols=["Quantity","UnitPrice"],
    outputCol="features"
)

data = assembler.transform(df_clean)
data.select("features","TotalAmount").show()

# Train Test Split

train_data, test_data = data.randomSplit([0.8,0.2],seed=42)

# Feature Scaling 

from pyspark.ml.feature import StandardScaler

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaledFeatures"
)

scaler_model = scaler.fit(train_data)

train_scaled = scaler_model.transform(train_data)
test_scaled = scaler_model.transform(test_data)

from pyspark.ml.regression import LinearRegression

lr = LinearRegression (
    featuresCol="scaledFeatures",
    labelCol="TotalAmount"
)

model = lr.fit(train_scaled)

# Model Prediction 

predictions = model.transform(test_scaled)

predictions.select(
    "Quantity","UnitPrice",
    "TotalAmount","prediction"
).show()

from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(
    labelCol="TotalAmount",
    predictionCol="prediction",metricName="rmse"
)

rmse = evaluator.evaluate(predictions)

print ("RMSE:",rmse)

print("Total Records:",df_clean.count())
print("Number of Partitions:",df_clean.rdd.getNumPartitions())

df_clean.filter(col("TotalAmount") < 500 )

from pyspark.ml.regression import RandomForestRegressor

rf = RandomForestRegressor(
    featuresCol="scaledFeatures",
    labelCol="TotalAmount"
)

rf_model = rf.fit(train_scaled)

rf_predictions = rf_model.transform(test_scaled)

rf_predictions.select(
    "Quantity",
    "UnitPrice",
    "TotalAmount",
    "prediction"
).show()

from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(
    labelCol="TotalAmount",
    predictionCol="prediction",
    metricName="rmse"
)

rf_rmse = evaluator.evaluate(rf_predictions)

print("Random Forest RMSE:", rf_rmse)

print(rf_model)

rf = RandomForestRegressor(
    featuresCol="scaledFeatures",
    labelCol="TotalAmount",
    numTrees=50
)

