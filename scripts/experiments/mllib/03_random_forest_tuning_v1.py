#!/usr/bin/env python3
import time
import random
import builtins
import itertools
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder \
    .appName("Random_Forest_Tuning") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

print("=== RANDOM FOREST TUNING (MLlib) ===")

# 1. Đọc dữ liệu, chia train/test 80/20 theo thời gian
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
    "numTrees": [100, 200, 300, 500],
    "maxDepth": [10, 20, 30],
    "minInstancesPerNode": [1, 2, 4],
    "featureSubsetStrategy": ["all", "sqrt", "0.7", "0.8"],
}
all_combos = list(itertools.product(*param_grid.values()))
N_ITER = 20
combos = random.Random(42).sample(all_combos, N_ITER)   # giống RandomizedSearchCV(n_iter=20)
print(f"Tổng tổ hợp: {len(all_combos)} | Thử ngẫu nhiên: {len(combos)}")

# 3. TimeSeriesSplit(n_splits=3) tự cài: cửa sổ train mở rộng dần
n_splits = 3
test_size = n_train // (n_splits + 1)
folds = []
for i in range(n_splits):
    a = n_train - (n_splits - i) * test_size
    folds.append((a, a + test_size))
print("Các fold:", folds)

mae_eval = RegressionEvaluator(labelCol="Revenue", predictionCol="prediction", metricName="mae")


def make_rf(trees, depth, leaf, fss):
    return RandomForestRegressor(featuresCol="features", labelCol="Revenue",
                                 numTrees=trees, maxDepth=depth,
                                 minInstancesPerNode=leaf,
                                 featureSubsetStrategy=fss,
                                 maxBins=32, seed=42)


# 4. Tìm kiếm tham số bằng CV theo thời gian (tiêu chí MAE)
t0 = time.time()
cv_results = []
for idx, (trees, depth, leaf, fss) in enumerate(combos, 1):
    fold_maes = []
    for a, b in folds:
        tr = train.filter(col("row_id") <= a)
        va = train.filter((col("row_id") > a) & (col("row_id") <= b))
        m = make_rf(trees, depth, leaf, fss).fit(tr)
        fold_maes.append(mae_eval.evaluate(m.transform(va)))
    cv_mae = builtins.sum(fold_maes) / len(fold_maes)
    cv_results.append((trees, depth, leaf, fss, float(cv_mae)))
    print(f"[{idx:>2}/{N_ITER}] numTrees={trees} | maxDepth={depth} | "
          f"minInst={leaf} | subset={fss} | CV MAE={cv_mae:,.2f}")

best = builtins.min(cv_results, key=lambda r: r[4])
b_trees, b_depth, b_leaf, b_fss, best_cv = best
search_time = time.time() - t0

print("\n=== HOÀN THÀNH TÌM KIẾM THAM SỐ ===")
print(f"Tham số tốt nhất: numTrees={b_trees}, maxDepth={b_depth}, "
      f"minInstancesPerNode={b_leaf}, featureSubsetStrategy={b_fss}")
print(f"MAE tốt nhất trên Cross-Validation: {best_cv:,.2f} GBP")

# 5. Huấn luyện lại trên toàn bộ train bằng tham số tốt nhất
start = time.time()
best_model = make_rf(b_trees, b_depth, b_leaf, b_fss).fit(train)
train_time = time.time() - start
predictions = best_model.transform(test)

# 6. Đánh giá
rmse = RegressionEvaluator(labelCol="Revenue", predictionCol="prediction", metricName="rmse").evaluate(predictions)
mae = mae_eval.evaluate(predictions)
r2 = RegressionEvaluator(labelCol="Revenue", predictionCol="prediction", metricName="r2").evaluate(predictions)

w = predictions.agg(sum(abs(col("Revenue") - col("prediction"))).alias("e"),
                    sum(abs(col("Revenue"))).alias("a")).first()
wape = w["e"] / w["a"] * 100

mape = predictions.filter(col("Revenue") > 0).select(
    avg(abs((col("Revenue") - col("prediction")) / col("Revenue"))).alias("m")
).first()["m"] * 100

print("\n=== KẾT QUẢ ĐÁNH GIÁ RANDOM FOREST (SAU KHI TUNING) ===")
print(f"MAE:  {mae:,.2f} GBP")
print(f"RMSE: {rmse:,.2f} GBP")
print(f"MAPE: {mape:.2f}%")
print(f"WAPE: {wape:.2f}%")
print(f"R2:   {r2:.4f}")
print(f"Thời gian tìm tham số: {search_time:.1f}s | Train lại: {train_time:.2f}s")

# 7. Ghi kết quả lên HDFS
MODEL_NAME = "random_forest_tuned_mllib_v1"
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

cv_df = spark.createDataFrame(
    cv_results,
    ["numTrees", "maxDepth", "minInstancesPerNode", "featureSubsetStrategy", "cv_mae"])
save_single_csv(cv_df, f"{RESULTS_DIR}/params/{MODEL_NAME}_cv.csv")

print(f"\nĐã lưu kết quả vào HDFS: {RESULTS_DIR}/metrics/{MODEL_NAME}.csv")
spark.stop()
