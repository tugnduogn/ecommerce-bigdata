#!/usr/bin/env python3

import pandas as pd
import numpy as np
import requests
from io import BytesIO
from sklearn.preprocessing import OneHotEncoder


# Cấu hình HDFS
HDFS_URL = "http://namenode:9870"
INPUT_PATH = "/ecommerce/gold/sales_forecast_raw"
OUTPUT_PATH = "/ecommerce/gold/daily_forecast_features_sklearn"


# Hàm đọc file từ HDFS bằng WebHDFS
def read_parquet_from_hdfs(hdfs_path):

    url = f"{HDFS_URL}/webhdfs/v1{hdfs_path}"

    # Lấy danh sách file trong thư mục HDFS
    response = requests.get(
        url,
        params={"op": "LISTSTATUS"}
    )

    response.raise_for_status()

    files = response.json()["FileStatuses"]["FileStatus"]

    parquet_files = [
        file["pathSuffix"]
        for file in files
        if file["pathSuffix"].endswith(".parquet")
    ]

    dataframes = []

    # Đọc từng file Parquet
    for file_name in parquet_files:

        file_path = f"{hdfs_path}/{file_name}"

        file_url = f"{HDFS_URL}/webhdfs/v1{file_path}"

        response = requests.get(
            file_url,
            params={"op": "OPEN"}
        )

        response.raise_for_status()

        df_part = pd.read_parquet(
            BytesIO(response.content)
        )

        dataframes.append(df_part)

    if not dataframes:
        raise ValueError("Không tìm thấy file Parquet trong HDFS!")

    return pd.concat(dataframes, ignore_index=True)


# Hàm lưu dữ liệu lên HDFS bằng WebHDFS
def write_parquet_to_hdfs(data, hdfs_path):

    # Xóa thư mục cũ nếu đã tồn tại
    url = f"{HDFS_URL}/webhdfs/v1{hdfs_path}"

    requests.delete(
        url,
        params={
            "op": "DELETE",
            "recursive": "true"
        }
    )

    # Tạo thư mục mới
    requests.put(
        url,
        params={"op": "MKDIRS"}
    ).raise_for_status()

    # Chuyển DataFrame thành Parquet trong bộ nhớ
    buffer = BytesIO()

    data.to_parquet(
        buffer,
        index=False,
        engine="pyarrow"
    )

    buffer.seek(0)

    # Tạo file Parquet trên HDFS
    file_path = f"{hdfs_path}/part-00000.parquet"

    file_url = f"{HDFS_URL}/webhdfs/v1{file_path}"

    response = requests.put(
        file_url,
        params={
            "op": "CREATE",
            "overwrite": "true",
            "noredirect": "true"
        }
    )

    response.raise_for_status()

    # Lấy URL của DataNode để upload dữ liệu
    upload_url = response.json()["Location"]

    upload_response = requests.put(
        upload_url,
        data=buffer.getvalue(),
        headers={
            "Content-Type": "application/octet-stream"
        }
    )

    upload_response.raise_for_status()


print("=== Chuẩn bị dữ liệu dự báo ===")


# Đọc dữ liệu từ HDFS
df = read_parquet_from_hdfs(INPUT_PATH)

print("Kích thước dữ liệu ban đầu:", df.shape)


# Chuyển đổi kiểu dữ liệu
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])


# Tính Revenue
df["Revenue"] = df["Quantity"] * df["Price"]


# Tổng hợp doanh thu theo ngày
daily = df.groupby(
    df["InvoiceDate"].dt.strftime("%Y-%m-%d")
)["Revenue"].sum().reset_index()

daily.columns = ["Date", "Revenue"]

daily["Date"] = pd.to_datetime(daily["Date"])

daily = daily.sort_values("Date").reset_index(drop=True)


# Tạo chuỗi ngày liên tục
date_range = pd.date_range(
    start=daily["Date"].min(),
    end=daily["Date"].max(),
    freq="D"
)

date_df = pd.DataFrame({
    "Date": date_range
})

daily = date_df.merge(
    daily,
    on="Date",
    how="left"
)

daily["Revenue"] = daily["Revenue"].fillna(0)


# Feature thời gian
daily["DayOfWeek"] = daily["Date"].dt.dayofweek
daily["Day"] = daily["Date"].dt.day
daily["Month"] = daily["Date"].dt.month
daily["Quarter"] = daily["Date"].dt.quarter
daily["Year"] = daily["Date"].dt.year

daily["IsWeekend"] = (
    daily["DayOfWeek"].isin([5, 6])
).astype(int)

daily["WeekOfYear"] = daily["Date"].dt.isocalendar().week.astype(int)

daily["DayName"] = daily["Date"].dt.day_name()

daily["YearMonth"] = daily["Date"].dt.strftime("%Y-%m")


# Lag Features
for lag_days in [1, 7, 14, 21, 28]:

    daily[f"Revenue_Lag_{lag_days}"] = (
        daily["Revenue"].shift(lag_days)
    )


# Rolling Mean & Std
for window in [7, 14, 30]:

    daily[f"Revenue_Rolling_Mean_{window}"] = (
        daily["Revenue"]
        .shift(1)
        .rolling(window=window)
        .mean()
    )

    daily[f"Revenue_Rolling_Std_{window}"] = (
        daily["Revenue"]
        .shift(1)
        .rolling(window=window)
        .std()
    )


# Lọc ngày bắt đầu giống Colab
daily = daily[
    daily["Date"] >= "2009-12-31"
].copy()


# One-Hot Encoding bằng Scikit-learn
encoder = OneHotEncoder(
    handle_unknown="error",
    drop=None,
    sparse_output=False,
    dtype=int
)

encoded_features = encoder.fit_transform(
    daily[["DayName", "YearMonth"]]
)

encoded_columns = encoder.get_feature_names_out(
    ["DayName", "YearMonth"]
)

encoded_df = pd.DataFrame(
    encoded_features,
    columns=encoded_columns,
    index=daily.index
)


# Ghép Feature đã Encoding vào dữ liệu
daily = pd.concat(
    [
        daily.drop(columns=["DayName", "YearMonth"]),
        encoded_df
    ],
    axis=1
)


print("SỐ CHIỀU DAYNAME:", len(encoder.categories_[0]))
print("SỐ CHIỀU YEARMONTH:", len(encoder.categories_[1]))

print("=== THỨ TỰ DAYNAME ===")
print(encoder.categories_[0])

print("=== THỨ TỰ YEARMONTH ===")
print(encoder.categories_[1])


# Xóa dòng thiếu
daily = daily.dropna().reset_index(drop=True)


print(f"Số dòng cuối cùng: {len(daily)}")
print("Kích thước dữ liệu:", daily.shape)

print("=== CẤU TRÚC DỮ LIỆU ===")
print(daily.dtypes)


# Lưu dữ liệu lên HDFS
write_parquet_to_hdfs(
    daily,
    OUTPUT_PATH
)


print("=== HOÀN THÀNH CHUẨN BỊ DỮ LIỆU ===")
print("Dữ liệu đã được lưu tại:", OUTPUT_PATH)
