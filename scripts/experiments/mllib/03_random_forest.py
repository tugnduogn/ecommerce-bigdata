#!/usr/bin/env python3
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder \
    .appName("Random_Forest") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

print("=== RANDOM FOREST REGRESSOR (THAM SỐ MẶC ĐỊNH) ===")

# 1. Đọc dữ liệu
df = spark.read.parquet("hdfs://namenode:9000/ecommerce/gold/daily_forecast_features")
df = df.orderBy("Date")

# 2. Khai báo Feature
feature_cols = [
    "DayOfWeek", "Day", "Month", "Quarter", "Year", "IsWeekend",
    "Revenue_Lag_1", "Revenue_Lag_7", "Revenue_Lag_14",
    "Revenue_Lag_21", "Revenue_Lag_28",
    "Revenue_Rolling_Mean_7", "Revenue_Rolling_Mean_14", "Revenue_Rolling_Mean_30",
    "Revenue_Rolling_Std_7", "Revenue_Rolling_Std_14", "Revenue_Rolling_Std_30",
    "DayName_vec", "YearMonth_vec"
]

data = VectorAssembler(inputCols=feature_cols, outputCol="features").transform(df)

# 3. Chia Train/Test theo thứ tự thời gian (80/20)
data = data.withColumn("row_id", row_number().over(Window.orderBy("Date")))
total = data.count()
split = int(total * 0.8)

train = data.filter(col("row_id") <= split)
test = data.filter(col("row_id") > split)

print("=== KIỂM TRA TRAIN/TEST ===")
print("Tổng số dòng:", total)
print("Train:", train.count(), "| Test:", test.count())

# 4. Khởi tạo Random Forest với tham số mặc định (chỉ cố định seed)
rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="Revenue",
    seed=42
)
# Giá trị mặc định của MLlib 3.0:
#   numTrees=20, maxDepth=5, maxBins=32, minInstancesPerNode=1,
#   minInfoGain=0.0, subsamplingRate=1.0, featureSubsetStrategy="auto"
#   (với hồi quy "auto" = onethird, tức mỗi lần chia xét 1/3 số đặc trưng)

# 5. Huấn luyện
start = time.time()
model = rf.fit(train)
train_time = time.time() - start

# 6. Dự đoán trên tập Test
predictions = model.transform(test)

# 7. Đánh giá
mae = RegressionEvaluator(labelCol="Revenue", predictionCol="prediction", metricName="mae").evaluate(predictions)
rmse = RegressionEvaluator(labelCol="Revenue", predictionCol="prediction", metricName="rmse").evaluate(predictions)
r2 = RegressionEvaluator(labelCol="Revenue", predictionCol="prediction", metricName="r2").evaluate(predictions)

w = predictions.agg(
    sum(abs(col("Revenue") - col("prediction"))).alias("total_error"),
    sum(abs(col("Revenue"))).alias("total_actual")
).first()
wape = w["total_error"] / w["total_actual"] * 100

mape = predictions.filter(col("Revenue") > 0).select(
    avg(abs((col("Revenue") - col("prediction")) / col("Revenue"))).alias("mape")
).first()["mape"] * 100

print("\n=== KẾT QUẢ ĐÁNH GIÁ RANDOM FOREST (THAM SỐ MẶC ĐỊNH) ===")
print(f"MAE:  {mae:,.2f} GBP")
print(f"RMSE: {rmse:,.2f} GBP")
print(f"MAPE: {mape:.2f}%")
print(f"WAPE: {wape:.2f}%")
print(f"R2:   {r2:.4f}")
print(f"Số cây: {model.getNumTrees} | Thời gian train: {train_time:.2f}s")

# 8. Ghi kết quả lên HDFS (/ecommerce/results)
MODEL_NAME = "random_forest_default_mllib"
RESULTS_DIR = "hdfs://namenode:9000/ecommerce/results"


def save_single_csv(sdf, final_path):
    """Ghi DataFrame thành MỘT file csv thay vì thư mục part-*."""
    tmp = final_path + "_tmp"
    sdf.coalesce(1).write.mode("overwrite").option("header", True).csv(tmp)
    Path = spark._jvm.org.apache.hadoop.fs.Path
    fs = Path(tmp).getFileSystem(spark._jsc.hadoopConfiguration())
    part = [s.getPath() for s in fs.listStatus(Path(tmp))
            if s.getPath().getName().startswith("part-")][0]
    dst = Path(final_path)
    if fs.exists(dst):
        fs.delete(dst, True)
    fs.rename(part, dst)
    fs.delete(Path(tmp), True)


metrics_df = spark.createDataFrame(
    [(MODEL_NAME, "MLlib", float(mae), float(rmse), float(mape),
      float(wape), float(r2), float(train_time))],
    ["model", "library", "MAE", "RMSE", "MAPE", "WAPE", "R2", "train_time_s"])
save_single_csv(metrics_df, f"{RESULTS_DIR}/metrics/{MODEL_NAME}.csv")

preds_df = (predictions
            .select(col("Date").cast("string").alias("Date"), "Revenue", "prediction")
            .withColumn("model", lit(MODEL_NAME))
            .orderBy("Date"))
save_single_csv(preds_df, f"{RESULTS_DIR}/predictions/{MODEL_NAME}.csv")

print(f"\nĐã lưu chỉ số vào HDFS: {RESULTS_DIR}/metrics/{MODEL_NAME}.csv")
print(f"Đã lưu dự đoán vào HDFS: {RESULTS_DIR}/predictions/{MODEL_NAME}.csv")

spark.stop()
