#!/usr/bin/env python3
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import lit
import time

spark = SparkSession.builder \
    .appName("Linear_Regression") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

print("=== Linear Regression ===")

df = spark.read.parquet("hdfs://namenode:9000/ecommerce/gold/daily_forecast_features")
df = df.orderBy("Date")

feature_cols = [
    "DayOfWeek", "Day", "Month", "Quarter", "Year", "IsWeekend",
    "Revenue_Lag_1", "Revenue_Lag_7", "Revenue_Lag_14",
    "Revenue_Lag_21", "Revenue_Lag_28",
    "Revenue_Rolling_Mean_7", "Revenue_Rolling_Mean_14", "Revenue_Rolling_Mean_30",
    "Revenue_Rolling_Std_7", "Revenue_Rolling_Std_14", "Revenue_Rolling_Std_30",
    "DayName_vec", "YearMonth_vec"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(df)

# Hiển thị 10 dòng dữ liệu đầu vào
print("\n=== 10 DÒNG DỮ LIỆU ĐẦU VÀO ===")

data.select(
    "Date",
    "Revenue",
    *feature_cols
).show(10, truncate=False)

w = Window.orderBy("Date")
data = data.withColumn("row_id", row_number().over(w))

total = data.count()
split = int(total * 0.8)

train = data.filter(col("row_id") <= split)
test = data.filter(col("row_id") > split)

print(f"Train: {train.count()} | Test: {test.count()}")
# Kiểm tra dữ liệu Train/Test
print("\n=== KIỂM TRA TRAIN/TEST UBUNTU ===")

print("Tổng số dòng:", total)
print("Train:", train.count())
print("Test:", test.count())

print("\n=== KHOẢNG THỜI GIAN TRAIN ===")
train.agg(
    min("Date").alias("Ngày bắt đầu"),
    max("Date").alias("Ngày kết thúc")
).show()

print("\n=== KHOẢNG THỜI GIAN TEST ===")
test.agg(
    min("Date").alias("Ngày bắt đầu"),
    max("Date").alias("Ngày kết thúc")
).show()

print("\n=== 5 DÒNG ĐẦU TRAIN ===")
train.orderBy("Date").select("Date", "Revenue").show(5)

print("\n=== 5 DÒNG ĐẦU TEST ===")
test.orderBy("Date").select("Date", "Revenue").show(5)
# Train
lr = LinearRegression(featuresCol="features", labelCol="Revenue")
start = time.time()
model = lr.fit(train)
train_time = time.time() - start
predictions = model.transform(test)

# Các chỉ số
evaluator_rmse = RegressionEvaluator(labelCol="Revenue", predictionCol="prediction", metricName="rmse")
evaluator_mae = RegressionEvaluator(labelCol="Revenue", predictionCol="prediction", metricName="mae")
evaluator_r2 = RegressionEvaluator(labelCol="Revenue", predictionCol="prediction", metricName="r2")

rmse = evaluator_rmse.evaluate(predictions)
mae = evaluator_mae.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

# MAPE và WAPE
predictions = predictions.withColumn("abs_error", abs(col("Revenue") - col("prediction")))
predictions = predictions.withColumn("ape", when(col("Revenue") != 0, col("abs_error") / col("Revenue")).otherwise(0))

mape = predictions.filter(col("Revenue") > 0).agg(avg("ape")).collect()[0][0] * 100
wape = predictions.agg(sum("abs_error")).collect()[0][0] / predictions.agg(sum(abs(col("Revenue")))).collect()[0][0] * 100

print("\n=== KẾT QUẢ LINEAR REGRESSION ===")
print(f"MAE:  {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"WAPE: {wape:.2f}%")
print(f"R2:   {r2:.4f}")
# ---------- Ghi kết quả lên HDFS (/ecommerce/results) ----------
MODEL_NAME = "linear_default_mllib"
RESULTS_DIR = "hdfs://namenode:9000/ecommerce/results"


def save_single_csv(sdf, final_path):
    """Ghi DataFrame thành MỘT file csv (giống sklearn), thay vì thư mục part-*."""
    tmp = final_path + "_tmp"
    sdf.coalesce(1).write.mode("overwrite").option("header", True).csv(tmp)

    jvm = spark._jvm
    conf = spark._jsc.hadoopConfiguration()
    Path = jvm.org.apache.hadoop.fs.Path
    fs = Path(tmp).getFileSystem(conf)

    part = [s.getPath() for s in fs.listStatus(Path(tmp))
            if s.getPath().getName().startswith("part-")][0]
    dst = Path(final_path)
    if fs.exists(dst):
        fs.delete(dst, True)
    fs.rename(part, dst)
    fs.delete(Path(tmp), True)


# Chỉ số (cùng cột với bản sklearn)
metrics_df = spark.createDataFrame(
    [(MODEL_NAME, "MLlib",
      float(mae), float(rmse), float(mape), float(wape), float(r2),
      float(train_time))],
    ["model", "library", "MAE", "RMSE", "MAPE", "WAPE", "R2", "train_time_s"]
)
save_single_csv(metrics_df, f"{RESULTS_DIR}/metrics/{MODEL_NAME}.csv")

# Dự đoán trên tập test
preds_df = (predictions
            .select(col("Date").cast("string").alias("Date"),
                    col("Revenue"),
                    col("prediction"))
            .withColumn("model", lit(MODEL_NAME))
            .orderBy("Date"))
save_single_csv(preds_df, f"{RESULTS_DIR}/predictions/{MODEL_NAME}.csv")

print(f"\nĐã lưu chỉ số vào HDFS: {RESULTS_DIR}/metrics/{MODEL_NAME}.csv")
print(f"Đã lưu dự đoán vào HDFS: {RESULTS_DIR}/predictions/{MODEL_NAME}.csv")

spark.stop()
spark.stop()
