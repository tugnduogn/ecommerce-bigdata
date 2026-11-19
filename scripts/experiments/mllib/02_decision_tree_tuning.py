#!/usr/bin/env python3
import time
import itertools
import builtins
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder \
    .appName("Decision_Tree_Tuning") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

print("=== DECISION TREE TUNING (MLlib) ===")

# 1. Đọc dữ liệu và chia train/test 80/20 theo thời gian
df = spark.read.parquet("hdfs://namenode:9000/ecommerce/gold/daily_forecast_features")

feature_cols = [
    "DayOfWeek", "Day", "Month", "Quarter", "Year", "IsWeekend",
    "Revenue_Lag_1", "Revenue_Lag_7", "Revenue_Lag_14",
    "Revenue_Lag_21", "Revenue_Lag_28",
    "Revenue_Rolling_Mean_7", "Revenue_Rolling_Mean_14", "Revenue_Rolling_Mean_30",
    "Revenue_Rolling_Std_7", "Revenue_Rolling_Std_14", "Revenue_Rolling_Std_30",
    "DayName_vec", "YearMonth_vec"
]

data = VectorAssembler(inputCols=feature_cols, outputCol="features").transform(df)
data = data.withColumn("row_id", row_number().over(Window.orderBy("Date")))

total = data.count()
split = int(total * 0.8)
train = data.filter(col("row_id") <= split).cache()
test = data.filter(col("row_id") > split).cache()
n_train = train.count()
print(f"Train: {n_train} | Test: {test.count()}")

# 2. Không gian tham số (khớp phần có thể khớp với sklearn)
param_grid = {
    "maxDepth": [2, 3, 4, 5, 6, 8],
    "minInstancesPerNode": [1, 5, 10, 20, 30],
}

combos = list(itertools.product(param_grid["maxDepth"],
                                param_grid["minInstancesPerNode"]))
print(f"Số tổ hợp tham số: {len(combos)}")

# 3. TimeSeriesSplit(n_splits=3) tự cài: cửa sổ train mở rộng dần
n_splits = 3
test_size = n_train // (n_splits + 1)
folds = []
for i in range(n_splits):
    test_start = n_train - (n_splits - i) * test_size
    folds.append((test_start, test_start + test_size))
print("Các fold (train <= a, val trong (a, b]):", folds)

mae_eval = RegressionEvaluator(labelCol="Revenue", predictionCol="prediction", metricName="mae")


def make_dt(depth, leaf):
    return DecisionTreeRegressor(featuresCol="features", labelCol="Revenue",
                                 maxDepth=depth, minInstancesPerNode=leaf,
                                 maxBins=32, seed=42)


# 4. Tìm kiếm tham số bằng CV theo thời gian (tiêu chí: MAE giống sklearn)
t0 = time.time()
cv_results = []
for depth, leaf in combos:
    fold_maes = []
    for a, b in folds:
        tr = train.filter(col("row_id") <= a)
        va = train.filter((col("row_id") > a) & (col("row_id") <= b))
        m = make_dt(depth, leaf).fit(tr)
        fold_maes.append(mae_eval.evaluate(m.transform(va)))
    cv_mae =builtins.sum(fold_maes) / len(fold_maes)
    cv_results.append((depth, leaf, cv_mae))
    print(f"maxDepth={depth:>2} | minInstancesPerNode={leaf} | CV MAE={cv_mae:,.2f}")

best_depth, best_leaf, best_cv = builtins.min(cv_results, key=lambda r: r[2])
search_time = time.time() - t0

print("\n=== HOÀN THÀNH TÌM KIẾM THAM SỐ ===")
print(f"Tham số tốt nhất: maxDepth={best_depth}, minInstancesPerNode={best_leaf}")
print(f"MAE tốt nhất trên Cross-Validation: {best_cv:,.2f} GBP")

# 5. Huấn luyện lại trên toàn bộ train bằng tham số tốt nhất, dự đoán test
start = time.time()
best_model = make_dt(best_depth, best_leaf).fit(train)
train_time = time.time() - start
predictions = best_model.transform(test)

# 6. Đánh giá (cùng công thức các bản khác)
rmse = RegressionEvaluator(labelCol="Revenue", predictionCol="prediction", metricName="rmse").evaluate(predictions)
mae = mae_eval.evaluate(predictions)
r2 = RegressionEvaluator(labelCol="Revenue", predictionCol="prediction", metricName="r2").evaluate(predictions)

w = predictions.agg(sum(abs(col("Revenue") - col("prediction"))).alias("e"),
                    sum(abs(col("Revenue"))).alias("a")).first()
wape = w["e"] / w["a"] * 100

mape = predictions.filter(col("Revenue") > 0).select(
    avg(abs((col("Revenue") - col("prediction")) / col("Revenue"))).alias("m")
).first()["m"] * 100

print("\n=== KẾT QUẢ ĐÁNH GIÁ DECISION TREE (SAU KHI TUNING) ===")
print(f"MAE:  {mae:,.2f} GBP")
print(f"RMSE: {rmse:,.2f} GBP")
print(f"MAPE: {mape:.2f}%")
print(f"WAPE: {wape:.2f}%")
print(f"R2:   {r2:.4f}")

# 7. Ghi kết quả lên HDFS
MODEL_NAME = "decision_tree_tuned_mllib"
RESULTS_DIR = "hdfs://namenode:9000/ecommerce/results"


def save_single_csv(sdf, final_path):
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
            .withColumn("model", lit(MODEL_NAME)).orderBy("Date"))
save_single_csv(preds_df, f"{RESULTS_DIR}/predictions/{MODEL_NAME}.csv")

# Bảng CV và tham số tốt nhất, để làm báo cáo
cv_df = spark.createDataFrame(cv_results, ["maxDepth", "minInstancesPerNode", "cv_mae"])
save_single_csv(cv_df, f"{RESULTS_DIR}/params/{MODEL_NAME}_cv.csv")

print(f"\nĐã lưu kết quả vào HDFS: {RESULTS_DIR}/metrics/{MODEL_NAME}.csv")
print(f"Thời gian tìm tham số: {search_time:.1f}s")
spark.stop()
