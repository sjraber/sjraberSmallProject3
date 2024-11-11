import time
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
import matplotlib.pyplot as plt
import pandas as pd

st = time.time()
#spark session
spark = SparkSession.builder.appName("FraudPrediction").getOrCreate()

#create spark dataframe
fraud = spark.read.csv("clean.csv", header=True, sep=",", inferSchema=True)
#fraud.printSchema()

#training and testing data
train = fraud.randomSplit([.5, .5])
test = fraud.randomSplit([.5, .5])
#tests
#train[0].show()
#test[1].show()


#features
featureCols = fraud.columns
assembler = VectorAssembler(inputCols=featureCols, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

#craeting mlp model layers
mlp = [len(featureCols), 10, 5, 2]

#declaring model and eveluators
mlpmodel = MultilayerPerceptronClassifier(featuresCol="scaledFeatures", labelCol="fraud", layers = mlp, blockSize = 128, maxIter = 100)
binary = BinaryClassificationEvaluator(labelCol="fraud")
multi = MulticlassClassificationEvaluator(labelCol="fraud", metricName="accuracy")

#making pipeline
pl = Pipeline(stages=[assembler, scaler, mlpmodel])

#fitting pipeline and doing predictions based on training
fittedModel = pl.fit(train[0])
predictions = fittedModel.transform(test[1])
predictions.select("scaledFeatures", "fraud", "prediction").show()
#predictions.where(predictions.fraud==1.0).show()

#declaring variables for doing precision/recall evaluations
truePositive = predictions.filter((col("fraud") == 1.0) & (col("prediction")==1.0)).count()
falsePositive = predictions.filter((col("fraud") == 0.0) & (col("prediction")==1.0)).count()
falseNegative = predictions.filter((col("fraud") == 1.0) & (col("prediction")==0.0)).count()

#doing evaluations
accuracy = multi.evaluate(predictions)
auc = binary.evaluate(predictions)
precision = truePositive / (truePositive + falsePositive)
recall = truePositive / (truePositive + falseNegative)

#showing evaluations
print(f"accuracy = {accuracy:.2f}")
print(f"AUC = {auc:.2f}")
print(f"precision = {precision:.2f}")
print(f"recall = {recall:.2f}")

et = time.time()
t = et - st
print(f"time to complete = {t:.2f}")
