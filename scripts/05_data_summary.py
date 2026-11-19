#!/usr/bin/env python3
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder \
    .appName("Data_Summary") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

print("=== Thống kê dữ liệu giao dịch ===")

# Đọc dữ liệu từ Silver
df = spark.read.parquet("hdfs://namenode:9000/ecommerce/silver/transactions_cleaned")
print(f"Số dòng từ Silver: {df.count()}")

# Danh sách StockCode cần loại bỏ
exclude_stockcodes = [
    'M', 'm',
    'D',
    'ADJUST', 'ADJUST2',
    'B',
    'S',
    'TEST001', 'TEST002',
    'C3', 'GIFT'
]

# 1. Loại bỏ các StockCode không hợp lệ
df = df.filter(~col("StockCode").isin(exclude_stockcodes))
print(f"Sau khi loại StockCode: {df.count()}")

# Danh sách StockCode cần loại bỏ đối với hóa đơn bắt đầu bằng C
exclude_credit_stockcodes = [
    'AMAZONFEE', 'BANK CHARGES',
    'CRUK', 'POST', 'DOT', 'C2'
]

# 2. Loại bỏ các giao dịch có Invoice bắt đầu bằng C
# và StockCode thuộc danh sách trên
df = df.filter(
    ~(
        col("Invoice").startswith("C") &
        col("StockCode").isin(exclude_credit_stockcodes)
    )
)
print(f"Sau khi lọc hóa đơn C và StockCode: {df.count()}")

# 3. Loại bỏ Quantity âm nếu Invoice KHÔNG bắt đầu bằng C
df = df.filter(
    (col("Quantity") >= 0) |
    col("Invoice").startswith("C")
)
print(f"Sau khi lọc Quantity: {df.count()}")

# 4. Chỉ giữ Price > 0
df = df.filter(col("Price") > 0)
print(f"Sau khi lọc Price > 0: {df.count()}")

# 5. Thống kê tổng số dòng dữ liệu sau khi lọc
total_rows = df.count()
print("===================================")
print(f"Tổng số dòng dữ liệu sau khi lọc: {total_rows}")
print("===================================")

# 6. Đẩy dữ liệu đã lọc lên HDFS
output_path = "hdfs://namenode:9000/ecommerce/gold/data_summary"

df.write.mode("overwrite").parquet(output_path)

print(f"Đã lưu dữ liệu lên HDFS: {output_path}")
print("=== Hoàn tất thống kê dữ liệu ===")

spark.stop()
