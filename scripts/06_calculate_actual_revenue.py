#!/usr/bin/env python3
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# =====================================================
# 0. CẤU HÌNH
# =====================================================

INPUT_PATH = "hdfs://namenode:9000/ecommerce/gold/data_summary"
OUTPUT_PATH = "hdfs://namenode:9000/ecommerce/results/actual_revenue"

# Các StockCode không phải sản phẩm, thống kê riêng như "chi phí mã đặc biệt".
# Viết HOA, vì StockCode sẽ được chuẩn hóa (upper + trim) trước khi so khớp.
TARGET_CODES = ["AMAZONFEE", "BANK CHARGES", "CRUK", "POST", "DOT", "C2"]

spark = (
    SparkSession.builder
    .appName("Calculate_Actual_Revenue")
    .master("spark://spark-master:7077")
    .getOrCreate()
)

print("=== TÍNH DOANH THU THỰC SỰ ===")


def save_csv(df, name):
    """Ghi 1 file CSV có header vào thư mục con của OUTPUT_PATH."""
    (
        df.coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv(f"{OUTPUT_PATH}/{name}")
    )


# =====================================================
# 1. ĐỌC VÀ CHUẨN HÓA DỮ LIỆU
# =====================================================

df = spark.read.parquet(INPUT_PATH)

df = (
    df
    # Chuẩn hóa để isin() không bị lệch do chữ hoa/thường hoặc khoảng trắng
    .withColumn("StockCode", F.upper(F.trim(F.col("StockCode").cast("string"))))
    .withColumn("Invoice", F.trim(F.col("Invoice").cast("string")))
    .withColumn("InvoiceDate", F.to_timestamp(F.col("InvoiceDate")))
    .withColumn("Date", F.to_date(F.col("InvoiceDate")))
    .withColumn("Amount", F.col("Quantity") * F.col("Price"))
    .withColumn("IsCancel", F.col("Invoice").startswith("C"))
    .withColumn("IsTarget", F.col("StockCode").isin(TARGET_CODES))
    .cache()
)

print(f"Số dòng dữ liệu đầu vào: {df.count()}")

# Kiểm tra nhanh các mã đặc biệt: nằm ở loại hóa đơn nào, dấu Quantity/Price ra sao
print("\n--- Phân bố các mã đặc biệt trong dữ liệu ---")
(
    df.filter("IsTarget")
    .groupBy(
        "StockCode",
        F.col("IsCancel").alias("Hoa_don_C"),
        (F.col("Quantity") > 0).alias("Quantity_duong"),
        (F.col("Price") > 0).alias("Price_duong"),
    )
    .count()
    .orderBy("StockCode", "Hoa_don_C")
    .show(50, truncate=False)
)

# =====================================================
# 2. DOANH THU BÁN SẢN PHẨM THEO NGÀY
# =====================================================

print("\n=== 1. DOANH THU BÁN SẢN PHẨM ===")

sales_daily = (
    df.filter(
        ~F.col("IsTarget")
        & ~F.col("IsCancel")
        & (F.col("Quantity") > 0)
        & (F.col("Price") > 0)
    )
    .groupBy("Date")
    .agg(F.sum("Amount").alias("Sales_Revenue"))
    .orderBy("Date")
)

print(f"Số ngày có dữ liệu doanh thu: {sales_daily.count()}")
sales_daily.show(20, truncate=False)
save_csv(sales_daily, "sales_revenue_daily")

# =====================================================
# 3. THỐNG KÊ ĐƠN HÀNG HOÀN THEO NGÀY
# =====================================================

print("\n=== 2. THỐNG KÊ ĐƠN HÀNG HOÀN ===")

returns_daily = (
    df.filter(
        ~F.col("IsTarget")
        & F.col("IsCancel")
        & (F.col("Quantity") < 0)
        & (F.col("Price") > 0)
    )
    .groupBy("Date")
    .agg(
        F.countDistinct("Invoice").alias("Return_Order_Count"),
        F.sum(F.abs(F.col("Amount"))).alias("Return_Amount"),
    )
    .orderBy("Date")
)

print(f"Số ngày có dữ liệu hoàn hàng: {returns_daily.count()}")
returns_daily.show(20, truncate=False)
save_csv(returns_daily, "returns_daily")

# =====================================================
# 4. CHI PHÍ CÁC MÃ STOCKCODE ĐẶC BIỆT THEO NGÀY
# =====================================================
# Lấy cả hóa đơn thường lẫn hóa đơn "C" (trước đây chỉ lấy hóa đơn "C"
# nên các mã như POST, DOT... ở hóa đơn thường bị mất hoàn toàn).
# Lấy trị tuyệt đối giá trị dòng; chỉ loại dòng có giá trị bằng 0.

print("\n=== 3. CHI PHÍ MÃ ĐẶC BIỆT ===")

target_daily = (
    df.filter(F.col("IsTarget") & (F.col("Amount") != 0))
    .groupBy("Date", "StockCode")
    .agg(
        F.sum(F.abs(F.col("Amount"))).alias("Target_Amount"),
        F.count("*").alias("Transaction_Count"),
    )
    .orderBy("Date", "StockCode")
)

print(f"Số dòng thống kê mã đặc biệt: {target_daily.count()}")
target_daily.show(30, truncate=False)
save_csv(target_daily, "target_expenses_daily")

target_expense_daily = target_daily.groupBy("Date").agg(
    F.sum("Target_Amount").alias("Target_Expense")
)

# =====================================================
# 5. DOANH THU THỰC SỰ THEO NGÀY
# =====================================================

print("\n=== 4. TÍNH DOANH THU THỰC SỰ ===")

actual_daily = (
    sales_daily
    .join(returns_daily.select("Date", "Return_Amount"), on="Date", how="full")
    .join(target_expense_daily, on="Date", how="full")
    .fillna(0, subset=["Sales_Revenue", "Return_Amount", "Target_Expense"])
    .withColumn(
        "Actual_Revenue",
        F.col("Sales_Revenue") - F.col("Return_Amount") - F.col("Target_Expense"),
    )
    .select("Date", "Sales_Revenue", "Return_Amount", "Target_Expense", "Actual_Revenue")
    .orderBy("Date")
    .cache()
)

print(f"Số ngày có dữ liệu doanh thu thực sự: {actual_daily.count()}")
actual_daily.show(30, truncate=False)
save_csv(actual_daily, "actual_revenue_daily")

# =====================================================
# 6. TỔNG HỢP THEO THÁNG VÀ NĂM
# =====================================================

SUM_COLUMNS = ["Sales_Revenue", "Return_Amount", "Target_Expense", "Actual_Revenue"]
aggregations = [F.sum(c).alias(c) for c in SUM_COLUMNS]

print("\n=== 5. TỔNG HỢP THEO THÁNG ===")

actual_monthly = (
    actual_daily
    .withColumn("YearMonth", F.date_format("Date", "yyyy-MM"))
    .groupBy("YearMonth")
    .agg(*aggregations)
    .orderBy("YearMonth")
)
actual_monthly.show(30, truncate=False)
save_csv(actual_monthly, "actual_revenue_monthly")

print("\n=== 6. TỔNG HỢP THEO NĂM ===")

actual_yearly = (
    actual_daily
    .withColumn("Year", F.year("Date"))
    .groupBy("Year")
    .agg(*aggregations)
    .orderBy("Year")
)
actual_yearly.show(20, truncate=False)
save_csv(actual_yearly, "actual_revenue_yearly")

print("\n=== HOÀN TẤT TÍNH DOANH THU THỰC SỰ ===")
print(f"Kết quả được lưu tại: {OUTPUT_PATH}")

spark.stop()
