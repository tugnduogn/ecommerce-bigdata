#!/usr/bin/env python3
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder \
    .appName("Prepare_Gold_Data") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

print("=== Chuẩn bị dữ liệu Gold ===")

# Đọc dữ liệu từ Silver
df = spark.read.parquet("hdfs://namenode:9000/ecommerce/silver/transactions_cleaned")
print(f"Số dòng từ Silver: {df.count()}")

# Danh sách StockCode cần loại bỏ
exclude_stockcodes = [
    'M', 'm',
    'POST', 'DOT', 'C2',
    'D',
    'ADJUST', 'ADJUST2',
    'AMAZONFEE',
    'BANK CHARGES',
    'B',
    'CRUK',
    'S',
    'TEST001', 'TEST002',
    'C3', 'GIFT'
]

# 1. Loại bỏ các StockCode không hợp lệ
df = df.filter(~col("StockCode").isin(exclude_stockcodes))
print(f"Sau khi loại StockCode: {df.count()}")

# 2. Xóa Quantity <= 0 nếu Invoice KHÔNG bắt đầu bằng 'C'
df = df.filter(
    (col("Quantity") > 0) | (col("Invoice").startswith("C"))
)
print(f"Sau khi lọc Quantity: {df.count()}")

# 3. Chỉ giữ Price > 0
df = df.filter(col("Price") > 0)
print(f"Sau khi lọc Price > 0: {df.count()}")

# ========== Dữ liệu cho RFM (xóa Customer_ID null) ==========
df_rfm = df.filter(col("Customer_ID").isNotNull())
print(f"Số dòng cho RFM: {df_rfm.count()}")

df_rfm.write.mode("overwrite") \
      .parquet("hdfs://namenode:9000/ecommerce/gold/customer_rfm_raw")

# ========== Dữ liệu cho dự báo (giữ Customer_ID null) ==========
df_forecast = df
print(f"Số dòng cho Forecast: {df_forecast.count()}")

df_forecast.write.mode("overwrite") \
           .parquet("hdfs://namenode:9000/ecommerce/gold/sales_forecast_raw")

print("=== Hoàn tất ===")
spark.stop()
