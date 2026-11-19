#!/usr/bin/env python3
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

spark = SparkSession.builder \
    .appName("Prepare_Forecast_Features") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

print("=== Chuẩn bị dữ liệu dự báo ===")

# Đọc dữ liệu
df = spark.read.parquet("hdfs://namenode:9000/ecommerce/gold/sales_forecast_raw")
df = df.withColumn("InvoiceDate", to_timestamp("InvoiceDate"))
df = df.withColumn("Revenue", col("Quantity") * col("Price"))

# Tổng hợp theo ngày
daily = df.groupBy(date_format("InvoiceDate", "yyyy-MM-dd").alias("Date")) \
           .agg(sum("Revenue").alias("Revenue")) \
           .orderBy("Date")

daily = daily.withColumn("Date", to_date("Date"))

# Tạo chuỗi ngày liên tục
min_date = daily.agg(min("Date")).collect()[0][0]
max_date = daily.agg(max("Date")).collect()[0][0]

date_df = spark.sql(f"""
    SELECT explode(sequence(to_date('{min_date}'), to_date('{max_date}'), interval 1 day)) as Date
""")

daily = date_df.join(daily, "Date", "left").fillna(0, subset=["Revenue"])

# Feature thời gian
daily =daily.withColumn("DayOfWeek", (dayofweek("Date") + 5) % 7) \
             .withColumn("Day", dayofmonth("Date")) \
             .withColumn("Month", month("Date")) \
             .withColumn("Quarter", quarter("Date")) \
             .withColumn("Year", year("Date")) \
             .withColumn("IsWeekend", when(dayofweek("Date").isin(1, 7), 1).otherwise(0)) \
             .withColumn("WeekOfYear", weekofyear("Date")) \
             .withColumn("DayName", date_format("Date", "EEEE")) \
             .withColumn("YearMonth", date_format("Date", "yyyy-MM"))

# Window
w = Window.orderBy("Date")

# Lag Features
for lag_days in [1, 7, 14, 21, 28]:
    daily = daily.withColumn(f"Revenue_Lag_{lag_days}", lag("Revenue", lag_days).over(w))

# Rolling Mean & Std
for window in [7, 14, 30]:
    w_roll = Window.orderBy("Date").rowsBetween(-window, -1)
    daily = daily.withColumn(f"Revenue_Rolling_Mean_{window}", avg("Revenue").over(w_roll))
    daily = daily.withColumn(f"Revenue_Rolling_Std_{window}", stddev("Revenue").over(w_roll))

# Lọc ngày bắt đầu giống Colab
daily = daily.filter(col("Date") >= "2009-12-31")

# One-Hot Encoding
indexer_day = StringIndexer(
    inputCol="DayName",
    outputCol="DayName_index",
    handleInvalid="error",
    stringOrderType="alphabetAsc"
)

encoder_day = OneHotEncoder(
    inputCol="DayName_index",
    outputCol="DayName_vec",
    dropLast=False
)

indexer_ym = StringIndexer(
    inputCol="YearMonth",
    outputCol="YearMonth_index",
    handleInvalid="error",
    stringOrderType="alphabetAsc"
)

encoder_ym = OneHotEncoder(
    inputCol="YearMonth_index",
    outputCol="YearMonth_vec",
    dropLast=False
)

pipeline = Pipeline(stages=[indexer_day, encoder_day, indexer_ym, encoder_ym])
pipeline_model = pipeline.fit(daily)
daily = pipeline_model.transform(daily)

print("SỐ CHIỀU DAYNAME:", daily.select("DayName_vec").first()[0].size)
print("SỐ CHIỀU YEARMONTH:", daily.select("YearMonth_vec").first()[0].size)
print("=== THỨ TỰ DAYNAME ===")
print(pipeline_model.stages[0].labels)

print("=== THỨ TỰ YEARMONTH ===")
print(pipeline_model.stages[2].labels)

# Xóa dòng thiếu
daily = daily.dropna()

print(f"Số dòng cuối cùng: {daily.count()}")
daily.printSchema()

# Lưu HDFS
daily.write.mode("overwrite").parquet("hdfs://namenode:9000/ecommerce/gold/daily_forecast_features")

print("=== Hoàn tất ===")
spark.stop()
