import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession

st.set_page_config(page_title="Ecommerce Analytics", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
.block-container { padding-top: 2rem; max-width: 1200px; }
[data-testid="stMetric"] {
    background: rgba(127,127,127,.08); border: 1px solid rgba(127,127,127,.2);
    border-left: 4px solid #3B4CCA; padding: 14px 18px; border-radius: 10px;
}
[data-testid="stMetricLabel"] p { font-size: .85rem; opacity: .75; }
[data-testid="stMetricValue"] { font-weight: 800; }
h1 { font-weight: 800; letter-spacing: -.02em; }
h2, h3 { font-weight: 700; }
</style>
""", unsafe_allow_html=True)

RESULTS = "hdfs://namenode:9000/ecommerce/results"
GOLD = "hdfs://namenode:9000/ecommerce/gold"


@st.cache_resource
def get_spark():
    return (SparkSession.builder.appName("Streamlit_HDFS")
            .master("local[*]").getOrCreate())


@st.cache_data(ttl=600)
def load_segments():
    return get_spark().read.parquet(f"{GOLD}/customer_segments").toPandas()


@st.cache_data(ttl=300)
def load_metrics():
    return (get_spark().read.csv(f"{RESULTS}/metrics/*.csv", header=True, inferSchema=True)
            .toPandas())


st.title("Ecommerce Analytics")
st.write("Hiểu khách hàng đang mua như thế nào và dự báo doanh thu những ngày tới. "
         "Chọn một trang bên dưới để bắt đầu.")

# ---------- Số liệu tổng quan ----------
seg, met = None, None
try:
    seg = load_segments()
except Exception:
    pass
try:
    met = load_metrics()
except Exception:
    pass

c1, c2, c3, c4 = st.columns(4)
if seg is not None:
    c1.metric("Khách hàng", f"{len(seg):,}")
    c2.metric("Nhóm khách hàng", seg["Cluster"].nunique())
    c3.metric("Tổng doanh thu", f"{seg['Monetary'].sum():,.0f}")
else:
    c1.metric("Khách hàng", "–")
    c2.metric("Nhóm khách hàng", "–")
    c3.metric("Tổng doanh thu", "–")

if met is not None and not met.empty:
    best = met.sort_values("MAE").iloc[0]
    c4.metric("Mô hình dự báo tốt nhất", best["model"], f"MAE {best['MAE']:,.0f}",
              delta_color="off")
else:
    c4.metric("Mô hình dự báo tốt nhất", "–")

if seg is None:
    st.warning("Chưa đọc được `gold/customer_segments` từ HDFS. Hãy chạy pipeline phân khúc trước.")
if met is None:
    st.info("Chưa có kết quả mô hình trong `/ecommerce/results`. Hãy chạy các script huấn luyện.")

# ---------- Điều hướng ----------
st.subheader("Các trang")
left, right = st.columns(2)
with left:
    st.markdown("#### Phân khúc khách hàng")
    st.write("Xem mỗi nhóm khách hàng chiếm bao nhiêu doanh thu, "
             "và nhóm nào mua gần đây, mua nhiều, chi tiêu lớn.")
    st.page_link("pages/1_Customer_Segments.py", label="Mở trang phân khúc", icon="👥")
with right:
    st.markdown("#### Dự báo doanh thu")
    st.write("So sánh các mô hình MLlib và scikit-learn trên cùng tập test, "
             "xem dự đoán so với doanh thu thực tế.")
    st.page_link("pages/2_Revenue_Forecast.py", label="Mở trang dự báo", icon="📈")

st.caption("Dữ liệu đọc từ HDFS qua Spark, lưu cache vài phút. Bấm phím C để xóa cache.")
