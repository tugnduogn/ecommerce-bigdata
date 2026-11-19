#!/usr/bin/env python3
import io
import time
import numpy as np
import pandas as pd
from hdfs import InsecureClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------- 1. Đọc parquet từ HDFS ----------
client = InsecureClient("http://namenode:9870", user="root")
HDFS_PATH = "/ecommerce/gold/daily_forecast_features_sklearn"


def read_parquet(path):
    files = [f for f in client.list(path) if f.endswith(".parquet")]
    frames = []
    for f in files:
        with client.read(f"{path}/{f}") as reader:
            frames.append(pd.read_parquet(io.BytesIO(reader.read())))
    return pd.concat(frames, ignore_index=True)


print("=== RANDOM FOREST TUNING (scikit-learn) ===")

df = read_parquet(HDFS_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# ---------- 2. Ma trận đặc trưng (cùng bộ với MLlib, không có WeekOfYear) ----------
numeric_cols = [
    "DayOfWeek", "Day", "Month", "Quarter", "Year", "IsWeekend",
    "Revenue_Lag_1", "Revenue_Lag_7", "Revenue_Lag_14",
    "Revenue_Lag_21", "Revenue_Lag_28",
    "Revenue_Rolling_Mean_7", "Revenue_Rolling_Mean_14", "Revenue_Rolling_Mean_30",
    "Revenue_Rolling_Std_7", "Revenue_Rolling_Std_14", "Revenue_Rolling_Std_30",
]
dayname_cols = sorted(c for c in df.columns if c.startswith("DayName_"))
yearmonth_cols = sorted(c for c in df.columns if c.startswith("YearMonth_"))
feature_names = numeric_cols + dayname_cols + yearmonth_cols

X = df[feature_names].to_numpy(dtype=float)
y = df["Revenue"].to_numpy(dtype=float)

# ---------- 3. Chia train/test 80/20 theo thời gian ----------
total = len(df)
split = int(total * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
dates = df["Date"]

print(f"Số đặc trưng: {X.shape[1]}")
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ---------- 4. Tìm siêu tham số (giống notebook Colab) ----------
param_dist_rf = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [1.0, 'sqrt', 0.7, 0.8]
}

tscv_rf = TimeSeriesSplit(n_splits=3)
rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)

random_search_rf = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_dist_rf,
    n_iter=20,
    scoring='neg_mean_absolute_error',
    cv=tscv_rf,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("=== BẮT ĐẦU TÌM KIẾM SIÊU THAM SỐ RANDOM FOREST ===")
t0 = time.time()
random_search_rf.fit(X_train, y_train)
search_time = time.time() - t0

best_rf_model = random_search_rf.best_estimator_

print("\n=== HOÀN THÀNH TÌM KIẾM THAM SỐ ===")
print("Tham số tốt nhất tìm được:", random_search_rf.best_params_)
print(f"MAE tốt nhất trên Cross-Validation: {-random_search_rf.best_score_:,.2f} GBP")

# ---------- 5. Dự đoán và đánh giá ----------
start = time.time()
best_rf_model.fit(X_train, y_train)      # train lại để đo thời gian
train_time = time.time() - start
y_pred = best_rf_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
wape = np.sum(np.abs(y_test - y_pred)) / np.sum(np.abs(y_test)) * 100
positive = y_test > 0
mape = np.mean(np.abs((y_test[positive] - y_pred[positive]) / y_test[positive])) * 100

print("\n=== KẾT QUẢ ĐÁNH GIÁ RANDOM FOREST (SAU KHI TUNING) ===")
print(f"MAE:  {mae:,.2f} GBP")
print(f"RMSE: {rmse:,.2f} GBP")
print(f"MAPE: {mape:.2f}%")
print(f"WAPE: {wape:.2f}%")
print(f"R2:   {r2:.4f}")
print(f"Thời gian tìm tham số: {search_time:.1f}s | Train lại: {train_time:.2f}s")

# 10 đặc trưng quan trọng nhất
imp = pd.Series(best_rf_model.feature_importances_, index=feature_names)
print("\n=== TOP 10 ĐẶC TRƯNG QUAN TRỌNG ===")
print(imp.sort_values(ascending=False).head(10).round(4).to_string())

# ---------- 6. Ghi kết quả lên HDFS (/ecommerce/results) ----------
MODEL_NAME = "random_forest_tuned_sklearn_v1"
RESULTS_DIR = "/ecommerce/results"

metrics = pd.DataFrame([{
    "model": MODEL_NAME, "library": "scikit-learn",
    "MAE": mae, "RMSE": rmse, "MAPE": mape, "WAPE": wape, "R2": r2,
    "train_time_s": train_time,
}])
client.makedirs(f"{RESULTS_DIR}/metrics")
client.write(f"{RESULTS_DIR}/metrics/{MODEL_NAME}.csv",
             data=metrics.to_csv(index=False), overwrite=True, encoding="utf-8")

preds = pd.DataFrame({
    "Date": dates.iloc[split:].dt.strftime("%Y-%m-%d").values,
    "Revenue": y_test,
    "prediction": y_pred,
    "model": MODEL_NAME,
})
client.makedirs(f"{RESULTS_DIR}/predictions")
client.write(f"{RESULTS_DIR}/predictions/{MODEL_NAME}.csv",
             data=preds.to_csv(index=False), overwrite=True, encoding="utf-8")

# Bảng CV của 20 tổ hợp đã thử, sắp theo MAE tốt nhất
cv = pd.DataFrame(random_search_rf.cv_results_)
cv = cv[["param_n_estimators", "param_max_depth", "param_min_samples_split",
         "param_min_samples_leaf", "param_max_features",
         "mean_test_score", "std_test_score", "rank_test_score"]].copy()
cv["cv_mae"] = -cv.pop("mean_test_score")
cv = cv.sort_values("rank_test_score").astype(str)
client.makedirs(f"{RESULTS_DIR}/params")
client.write(f"{RESULTS_DIR}/params/{MODEL_NAME}_cv.csv",
             data=cv.to_csv(index=False), overwrite=True, encoding="utf-8")

print(f"\nĐã lưu chỉ số vào HDFS: {RESULTS_DIR}/metrics/{MODEL_NAME}.csv")
print(f"Đã lưu dự đoán vào HDFS: {RESULTS_DIR}/predictions/{MODEL_NAME}.csv")
print(f"Đã lưu bảng CV vào HDFS: {RESULTS_DIR}/params/{MODEL_NAME}_cv.csv")
