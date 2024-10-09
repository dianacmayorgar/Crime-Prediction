from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, sum
from pyspark.sql import functions as F

# Start a Spark Session
spark = SparkSession.builder.appName('Analysis_Robbery_Assault_Toronto').getOrCreate()

# Load the data
df_robbery = spark.read.csv('/Robbery_Clean.csv', header=True, inferSchema=True)
df_assault = spark.read.csv('/Assault_Clean.csv', header=True, inferSchema=True)

# Merge the dataframes
df_combined = df_robbery.union(df_assault)

# Statistical summaries
df_descriptive = df_combined.describe()
df_descriptive.show()

# Define the conditions for counting Assault and Robbery cases
assault_cond = when(col("MCI_CATEGORY") == "Assault", 1).otherwise(0)
robbery_cond = when(col("MCI_CATEGORY") == "Robbery", 1).otherwise(0)

# Perform the aggregation
df_counts = df_combined.withColumn("Assault", assault_cond) \
    .withColumn("Robbery", robbery_cond) \
    .groupBy("NEIGHBOURHOOD_158") \
    .agg(
        sum("Assault").alias("Assault"),
        sum("Robbery").alias("Robbery")
    ) \
    .withColumn("Total", col("Assault") + col("Robbery")) \
    .orderBy("Total", ascending=False)

# Aggregations, such as counting the number of robberies and assaults per neighborhood
# For MCI_CATEGORY "Assault"
top10_assault = df_combined.filter(df_combined.MCI_CATEGORY == 'Assault') \
    .groupBy('NEIGHBOURHOOD_158').count() \
    .orderBy('count', ascending=False) \
    .limit(10)

top10_assault.show()

# For MCI_CATEGORY "Robbery"
top10_robbery = df_combined.filter(df_combined.MCI_CATEGORY == 'Robbery') \
    .groupBy('NEIGHBOURHOOD_158').count() \
    .orderBy('count', ascending=False) \
    .limit(10)

top10_robbery.show()

# Most common hour for crimes
df_common_hour = df_combined.groupBy('OCC_HOUR').count().orderBy('count', ascending=False)
df_common_hour.show()

# Aggregations, most common hour for crimes 
# For MCI_CATEGORY "Assault"
common_hour_assault = df_combined.filter(df_combined.MCI_CATEGORY == 'Assault') \
    .groupBy('OCC_HOUR').count() \
    .orderBy('count', ascending=False) \
    .limit(10)

common_hour_assault.show()

# For MCI_CATEGORY "Robbery"
common_hour_robbery = df_combined.filter(df_combined.MCI_CATEGORY == 'Robbery') \
    .groupBy('OCC_HOUR').count() \
    .orderBy('count', ascending=False) \
    .limit(10)

common_hour_robbery.show()


# Calculate average and standard deviation for a numerical column, for example, 'OCC_HOUR'
df_combined.groupBy('NEIGHBOURHOOD_158').agg(
    F.avg('OCC_HOUR').alias('average_hour'),
    F.stddev('OCC_HOUR').alias('stddev_hour')
).show()

# Count the number of crimes for each day of the week
df_by_dayofweek = df_combined.groupBy('OCC_DOW').count().orderBy('count', ascending=False)
df_by_dayofweek.show()

# For MCI_CATEGORY "Assault"
day_assault = df_combined.filter(df_combined.MCI_CATEGORY == 'Assault') \
    .groupBy('OCC_DOW').count() \
    .orderBy('count', ascending=False) \
    .limit(10)

day_assault.show()

# For MCI_CATEGORY "Robbery"
day_robbery = df_combined.filter(df_combined.MCI_CATEGORY == 'Robbery') \
    .groupBy('OCC_DOW').count() \
    .orderBy('count', ascending=False) \
    .limit(10)

day_robbery.show()

# Trend of crimes by year
df_trend_by_year = df_combined.groupBy('OCC_YEAR').count().orderBy('OCC_YEAR', ascending=False)
df_trend_by_year.show()

# Trend of crimes by month
df_trend_by_month = df_combined.groupBy('OCC_MONTH').count().orderBy('count', ascending=False)
df_trend_by_month.show()

# For MCI_CATEGORY "Assault"
month_assault = df_combined.filter(df_combined.MCI_CATEGORY == 'Assault') \
    .groupBy('OCC_MONTH').count() \
    .orderBy('count', ascending=False) \
    .limit(10)

month_assault.show()

# For MCI_CATEGORY "Robbery"
month_robbery = df_combined.filter(df_combined.MCI_CATEGORY == 'Robbery') \
    .groupBy('OCC_MONTH').count() \
    .orderBy('count', ascending=False) \
    .limit(10)

month_robbery.show()

# Count of each type of crime by premises type
df_crime_by_location = df_combined.groupBy('PREMISES_TYPE', 'OFFENCE').count().orderBy('count', ascending=False)
df_crime_by_location.show(10)

"""PROBABILITY OF CRIME OCCURRENCE"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType
from pyspark.ml.feature import StringIndexer

# Categorize hours
def categorizar_hora(hora):
    if 0 <= hora < 6:
        return 'early morning'
    elif 6 <= hora < 12:
        return 'morning'
    elif 12 <= hora < 18:
        return 'afternoon'
    else:
        return 'night'

categorizar_hora_udf = udf(categorizar_hora, StringType())

df_combined = df_combined.withColumn('HOUR_CATEGORY', categorizar_hora_udf(df_combined['OCC_HOUR']))

# Index string columns
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df_combined) for column in ['OCC_MONTH', 'OCC_DOW', 'HOUR_CATEGORY']]
for indexer in indexers:
    df_combined = indexer.transform(df_combined)

df_combined.printSchema()

df_combined.coalesce(1).write.csv("/df_combined.csv", header=True)

data_types = df_combined.dtypes

for column, dtype in data_types:
    print(f"Columna: {column}, Tipo de dato: {dtype}")

crime_counts = df_combined.groupBy('OCC_MONTH', 'OCC_MONTH_index', 'OCC_DOW', 'OCC_DOW_index', 'HOUR_CATEGORY_index', 'HOUR_CATEGORY', 'NEIGHBOURHOOD_158', 'HOOD_158', 'MCI_CATEGORY').count()

total_crimes_by_neighbourhood = df_combined.groupBy('HOOD_158').count().withColumnRenamed('count', 'total_in_hood')

crime_probability = crime_counts.join(total_crimes_by_neighbourhood, 'HOOD_158')

crime_probability = crime_probability.withColumn('probability', col('count') / col('total_in_hood'))

crime_probability.show()

"""PREDICTION"""

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

feature_cols = ['OCC_MONTH_index', 'OCC_DOW_index', 'HOUR_CATEGORY_index', 'HOOD_158']  # Asegúrate de que estas columnas están indexadas correctamente
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# regression model
rf = RandomForestRegressor(labelCol="count", featuresCol="features")

# pipeline
pipeline = Pipeline(stages=[assembler, rf])

paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 20, 30]) \
    .build()

(train_data, test_data) = crime_probability.randomSplit([0.7, 0.3])

train_data.printSchema()

train_data.show()

cv = CrossValidator(estimator=pipeline,
                    estimatorParamMaps=paramGrid,
                    evaluator=RegressionEvaluator(labelCol="count"),
                    numFolds=5)

# training
cvModel = cv.fit(train_data)

# predictions
predictions = cvModel.transform(test_data)

# Errors
evaluator = RegressionEvaluator(labelCol="count", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

predictions.show()

predictions_to_save = predictions.select(
    "HOOD_158", "OCC_MONTH", "OCC_DOW", "HOUR_CATEGORY", 'NEIGHBOURHOOD_158', 'MCI_CATEGORY', "count", "prediction"
)

predictions_to_save.coalesce(1).write.csv("/predictions.csv", header=True)