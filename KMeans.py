from pyspark.ml.clustering import KMeans
from pyspark.sql import SparkSession

# creating sparksession and giving an app name
spark = SparkSession.builder.appName('KMeans Implementation').getOrCreate()

# Loads data.
dataset = spark.read.format("libsvm").load("/home/hadoop/workspacepython/kmeans_input.txt")

# Trains a k-means model.
kmeans = KMeans(k=2, seed=1)
model = kmeans.fit(dataset)

# Make predictions
predictions = model.transform(dataset)

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)
