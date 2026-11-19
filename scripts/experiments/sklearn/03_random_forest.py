#!/usr/bin/env python3
import io
import time
import numpy as np
import pandas as pd
from hdfs import InsecureClient
from sklearn.ensemble import RandomForestRegressor
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


print("=== Random Forest (scikit-learn, tham số mặc định) ===")

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

print(f"Số dòng: {len(df)} | Số đặc trưng: {X.shape[1]}")

# ---------- 3. Chia train/test 80/20 theo thời gian ----------
total = len(df)
split = int(total * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
dates = df["Date"]

print("\n=== KIỂM TRA TRAIN/TEST ===")
print("Tổng số dòng:", total)
print("Train:", len(X_train), "| Test:", len(X_test))
print("Khoảng thời gian train:", dates.iloc[0].date(), "->", dates.iloc[split - 1].date())
print("Khoảng thời gian test: ", dates.iloc[split].date(), "->", dates.iloc[-1].date())

# ---------- 4. Huấn luyện (siêu tham số mặc định) ----------
start = time.time()
model = RandomForestRegressor(random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
train_time = time.time() - start

y_pred = model.predict(X_test)

# ---------- 5. Đánh giá ----------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
wape = np.sum(np.abs(y_test - y_pred)) / np.sum(np.abs(y_test)) * 100

positive = y_test > 0
mape = np.mean(np.abs((y_test[positive] - y_pred[positive]) / y_test[positive])) * 100

print("\n=== KẾT QUẢ RANDOM FOREST (THAM SỐ MẶC ĐỊNH) ===")
print(f"MAE:  {mae:,.2f} GBP")
print(f"RMSE: {rmse:,.2f} GBP")
print(f"MAPE: {mape:.2f}%")
print(f"WAPE: {wape:.2f}%")
print(f"R2:   {r2:.4f}")
print(f"Số cây: {model.n_estimators} | Thời gian train: {train_time:.3f}s")

# ---------- 6. Ghi kết quả lên HDFS (/ecommerce/results) ----------
MODEL_NAME = "random_forest_default_sklearn"
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

print(f"\nĐã lưu chỉ số vào HDFS: {RESULTS_DIR}/metrics/{MODEL_NAME}.csv")
print(f"Đã lưu dự đoán vào HDFS: {RESULTS_DIR}/predictions/{MODEL_NAME}.csv")
