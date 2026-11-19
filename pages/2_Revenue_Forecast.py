import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pyspark.sql import SparkSession

st.set_page_config(page_title="Dự báo doanh thu", layout="wide")
st.title("So sánh mô hình dự báo doanh thu theo ngày")

RESULTS = "hdfs://namenode:9000/ecommerce/results"


@st.cache_resource
def get_spark():
    return (SparkSession.builder
            .appName("Streamlit_Results")
            .master("local[*]")
            .getOrCreate())


@st.cache_data(ttl=300)
def load_metrics():
    spark = get_spark()
    return (spark.read.csv(f"{RESULTS}/metrics/*.csv", header=True, inferSchema=True)
            .toPandas())

@st.cache_data(ttl=300)
def load_predictions():
    spark = get_spark()

    df = (
        spark.read.csv(
            f"{RESULTS}/predictions/*.csv",
            header=True,
            inferSchema=False
        )
        .toPandas()
    )

    # Chuyển đổi kiểu dữ liệu
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
    df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")

    return df.sort_values(["model", "Date"])

try:
    metrics = load_metrics()
    preds = load_predictions()
except Exception as e:
    st.error("Chưa đọc được kết quả từ HDFS. Hãy chạy các script huấn luyện trước.")
    st.exception(e)
    st.stop()

# ---------------- Bộ lọc ----------------
st.sidebar.header("Bộ lọc")
libs = sorted(metrics["library"].unique())
sel_libs = st.sidebar.multiselect("Thư viện", libs, default=libs)
m = metrics[metrics["library"].isin(sel_libs)]

all_models = sorted(m["model"].unique())
sel_models = st.sidebar.multiselect("Mô hình", all_models, default=all_models)
m = m[m["model"].isin(sel_models)]

if m.empty:
    st.warning("Hãy chọn ít nhất một mô hình.")
    st.stop()

sort_by = st.sidebar.selectbox("Xếp hạng theo", ["MAE", "RMSE", "MAPE", "WAPE", "R2"])
ascending = sort_by != "R2"          # R2 càng cao càng tốt
m = m.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

# ---------------- Chỉ số tổng quan ----------------
best = m.iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Mô hình tốt nhất", best["model"])
c2.metric("MAE", f"{best['MAE']:,.0f}")
c3.metric("WAPE", f"{best['WAPE']:.2f}%")
c4.metric("R²", f"{best['R2']:.4f}")

# ---------------- Bảng so sánh ----------------
st.subheader("Bảng so sánh chỉ số")
cols = ["model", "library", "MAE", "RMSE", "MAPE", "WAPE", "R2", "train_time_s"]
st.dataframe(
    m[cols].style
        .format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "MAPE": "{:.2f}",
                 "WAPE": "{:.2f}", "R2": "{:.4f}", "train_time_s": "{:.2f}"})
        .highlight_min(subset=["MAE", "RMSE", "MAPE", "WAPE"], color="#c6efce")
        .highlight_max(subset=["R2"], color="#c6efce"),
    use_container_width=True,
)

# ---------------- Biểu đồ so sánh ----------------
st.subheader("So sánh mô hình")
metric_choice = st.radio("Chỉ số", ["MAE", "RMSE", "WAPE", "R2"], horizontal=True)
fig_bar = px.bar(m, x="model", y=metric_choice, color="library",
                 text_auto=".3s", title=f"{metric_choice} theo mô hình")
fig_bar.update_layout(xaxis_title="", xaxis_tickangle=-30)
st.plotly_chart(fig_bar, use_container_width=True)

# ---------------- Thực tế vs dự đoán ----------------
st.subheader("Doanh thu thực tế và dự đoán (tập test)")
pm = preds[preds["model"].isin(sel_models)]
chosen = st.multiselect("Chọn mô hình để vẽ", sel_models,
                        default=sel_models[:2] if len(sel_models) > 1 else sel_models)

if chosen:
    actual = pm[pm["model"] == chosen[0]][["Date", "Revenue"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual["Date"], y=actual["Revenue"],
                             name="Thực tế", line=dict(color="black", width=2)))
    for name in chosen:
        d = pm[pm["model"] == name]
        fig.add_trace(go.Scatter(x=d["Date"], y=d["prediction"], name=name,
                                 mode="lines", line=dict(dash="dot")))
    fig.update_layout(xaxis_title="Ngày", yaxis_title="Doanh thu (GBP)",
                      hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Sai số theo ngày
    err = pm[pm["model"].isin(chosen)].copy()
    err["abs_error"] = (err["Revenue"] - err["prediction"]).abs()
    fig_err = px.line(err, x="Date", y="abs_error", color="model",
                      title="Sai số tuyệt đối theo ngày")
    st.plotly_chart(fig_err, use_container_width=True)

st.caption("Dữ liệu đọc từ HDFS: /ecommerce/results/metrics và /predictions "
           "(cache 5 phút, bấm C để xóa cache).")
