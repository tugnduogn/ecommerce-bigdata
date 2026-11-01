#!/usr/bin/env python3
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder \
    .appName("Ecommerce_ETL_Cleaning") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("=== Bắt đầu làm sạch dữ liệu ===")

df = spark.read.option("header", True) \
               .option("inferSchema", True) \
               .csv("hdfs://namenode:9000/ecommerce/bronze/online_retail_II.csv")

print(f"Số dòng ban đầu: {df.count()}")

# Chuẩn hóa tên cột
df = df.toDF(*[c.strip().replace(" ", "_") for c in df.columns])

# Xóa dòng trùng lặp
df = df.dropDuplicates()
print(f"Số dòng sau khi xóa trùng lặp: {df.count()}")

# Chuyển ngày tháng sang datetime
df = df.withColumn("InvoiceDate", to_timestamp("InvoiceDate", "dd/MM/yyyy HH:mm"))

# Đếm số dòng ngày tháng lỗi
invalid_date_count = df.filter(col("InvoiceDate").isNull()).count()
print(f"Số dòng có ngày tháng không hợp lệ: {invalid_date_count}")

# Ghi ra Silver
df.write.mode("overwrite") \
  .parquet("hdfs://namenode:9000/ecommerce/silver/transactions_cleaned")

print("=== Làm sạch hoàn tất ===")
spark.stop()
