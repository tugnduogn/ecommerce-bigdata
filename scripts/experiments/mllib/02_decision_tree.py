#!/usr/bin/env python3

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import lit
import time

spark = SparkSession.builder \
    .appName("Decision_Tree") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

print("=== DECISION TREE REGRESSOR ===")

# 1. Đọc dữ liệu
df = spark.read.parquet(
    "hdfs://namenode:9000/ecommerce/gold/daily_forecast_features"
)

df = df.orderBy("Date")

# 2. Khai báo Feature
feature_cols = [
    "DayOfWeek", "Day", "Month", "Quarter", "Year", "IsWeekend",
    "Revenue_Lag_1", "Revenue_Lag_7", "Revenue_Lag_14",
    "Revenue_Lag_21", "Revenue_Lag_28",
    "Revenue_Rolling_Mean_7", "Revenue_Rolling_Mean_14",
    "Revenue_Rolling_Mean_30",
    "Revenue_Rolling_Std_7", "Revenue_Rolling_Std_14",
    "Revenue_Rolling_Std_30",
    "DayName_vec", "YearMonth_vec"
]

# 3. Kết hợp Feature
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

data = assembler.transform(df)

# 4. Chia Train/Test theo thứ tự thời gian (80/20)
w = Window.orderBy("Date")

data = data.withColumn(
    "row_id",
    row_number().over(w)
)

total = data.count()
split = int(total * 0.8)

train = data.filter(col("row_id") <= split)
test = data.filter(col("row_id") > split)

print("=== KIỂM TRA TRAIN/TEST ===")
print("Tổng số dòng:", total)
print("Train:", train.count())
print("Test:", test.count())

# 5. Khởi tạo Decision Tree với tham số mặc định
dt = DecisionTreeRegressor(
    featuresCol="features",
    labelCol="Revenue",
    seed=42
)

# 6. Huấn luyện mô hình
start = time.time()
model = dt.fit(train)
train_time = time.time() - start

# 7. Dự đoán trên tập Test
predictions = model.transform(test)

# 8. Đánh giá mô hình
mae_evaluator = RegressionEvaluator(
    labelCol="Revenue",
    predictionCol="prediction",
    metricName="mae"
)

rmse_evaluator = RegressionEvaluator(
    labelCol="Revenue",
    predictionCol="prediction",
    metricName="rmse"
)

r2_evaluator = RegressionEvaluator(
    labelCol="Revenue",
    predictionCol="prediction",
    metricName="r2"
)

mae = mae_evaluator.evaluate(predictions)
rmse = rmse_evaluator.evaluate(predictions)
r2 = r2_evaluator.evaluate(predictions)

# Tính WAPE
wape_values = predictions.agg(
    sum(abs(col("Revenue") - col("prediction"))).alias("total_error"),
    sum(abs(col("Revenue"))).alias("total_actual")
).first()

wape = wape_values["total_error"] / wape_values["total_actual"] * 100

# Tính MAPE trên những ngày thực tế có doanh thu dương
positive_predictions = predictions.filter(col("Revenue") > 0)

mape = positive_predictions.select(
    avg(
        abs(
            (col("Revenue") - col("prediction")) / col("Revenue")
        )
    ).alias("mape")
).first()["mape"] * 100

# 9. In kết quả
print("=== KẾT QUẢ ĐÁNH GIÁ DECISION TREE ===")
print(f"MAE:  {mae:,.2f} GBP")
print(f"RMSE: {rmse:,.2f} GBP")
print(f"MAPE: {mape:.2f}%")
print(f"WAPE: {wape:.2f}%")
print(f"R2:   {r2:.4f}")

# ---------- Ghi kết quả lên HDFS (/ecommerce/results) ----------
MODEL_NAME = "decision_tree_default_mllib"
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


# Chỉ số (cùng cột với bản sklearn và Linear MLlib)
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
