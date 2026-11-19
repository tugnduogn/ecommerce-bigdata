#!/usr/bin/env python3
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType

spark = SparkSession.builder \
    .appName("RFM_KMeans") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("=== Bắt đầu tính RFM + KMeans ===")

# Đọc dữ liệu
df = spark.read.parquet("hdfs://namenode:9000/ecommerce/gold/customer_rfm_raw")

max_date = df.select(F.max("InvoiceDate")).collect()[0][0]
print(f"Ngày reference: {max_date}")

# Tính RFM
rfm = df.groupBy("Customer_ID").agg(
    datediff(lit(max_date), max("InvoiceDate")).alias("Recency"),
    countDistinct("Invoice").alias("Frequency"),
    sum("Revenue").alias("Monetary")
).na.fill(0)

print(f"Số khách hàng: {rfm.count()}")

# Chuẩn hóa dữ liệu
assembler = VectorAssembler(
    inputCols=["Recency", "Frequency", "Monetary"],
    outputCol="features"
)
rfm_vec = assembler.transform(rfm)

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features",
    withStd=True,
    withMean=True
)
rfm_scaled = scaler.fit(rfm_vec).transform(rfm_vec)

# KMeans với k=5
kmeans = KMeans(featuresCol="scaled_features", k=5, seed=42)
model = kmeans.fit(rfm_scaled)
predictions = model.transform(rfm_scaled)

# Lưu kết quả
result = predictions.select("Customer_ID", "Recency", "Frequency", "Monetary", "prediction") \
                   .withColumnRenamed("prediction", "Cluster")

result.write.mode("overwrite") \
      .parquet("hdfs://namenode:9000/ecommerce/gold/customer_segments")

print("=== Hoàn tất RFM + KMeans ===")
spark.stop()
