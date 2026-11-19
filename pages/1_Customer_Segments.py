import streamlit as st
import pandas as pd
import plotly.express as px
from pyspark.sql import SparkSession

st.set_page_config(page_title="Customer Segments", page_icon="👥", layout="wide")

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

PALETTE = ["#3B4CCA", "#0F9D8A", "#E5A400", "#D6456B",
           "#7B5CD6", "#2E86C1", "#8D6E63", "#546E7A"]


@st.cache_resource
def get_spark():
    return (SparkSession.builder.appName("Streamlit_HDFS")
            .master("local[*]").getOrCreate())


@st.cache_data(ttl=600)
def load_data():
    df = (get_spark().read
          .parquet("hdfs://namenode:9000/ecommerce/gold/customer_segments")
          .toPandas())
    df["Cluster"] = df["Cluster"].astype(str)   # để mỗi nhóm một màu rời
    return df


st.title("Phân khúc khách hàng")
st.write("Khách hàng được chia nhóm theo ba tiêu chí RFM: mua gần đây (Recency), "
         "mua thường xuyên (Frequency) và chi tiêu (Monetary).")

try:
    df = load_data()
except Exception as e:
    st.error("Chưa đọc được dữ liệu phân khúc từ HDFS. "
             "Kiểm tra đường dẫn `gold/customer_segments` và HDFS đang chạy.")
    st.exception(e)
    st.stop()

clusters_all = sorted(df["Cluster"].unique(), key=lambda x: int(x) if x.isdigit() else x)
color_map = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(clusters_all)}

# ---------- Bộ lọc ----------
st.sidebar.header("Bộ lọc")
clusters = st.sidebar.multiselect("Nhóm khách hàng", clusters_all, default=clusters_all)
rmin, rmax = int(df["Recency"].min()), int(df["Recency"].max())
rec_range = st.sidebar.slider("Số ngày từ lần mua cuối", rmin, rmax, (rmin, rmax))

dff = df[df["Cluster"].isin(clusters) & df["Recency"].between(*rec_range)]
if dff.empty:
    st.warning("Không có khách hàng nào khớp bộ lọc. Hãy chọn thêm nhóm hoặc nới khoảng ngày.")
    st.stop()

# ---------- Chỉ số ----------
total_rev = dff["Monetary"].sum()
rev_by = dff.groupby("Cluster")["Monetary"].sum()
top_cluster = rev_by.idxmax()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Khách hàng", f"{len(dff):,}")
c2.metric("Chi tiêu trung bình", f"{dff['Monetary'].mean():,.0f}")
c3.metric("Tổng doanh thu", f"{total_rev:,.0f}")
c4.metric("Nhóm doanh thu cao nhất", f"Nhóm {top_cluster}",
          f"{rev_by.max() / total_rev:.0%} doanh thu", delta_color="off")

st.write("")

# ---------- Số lượng và doanh thu ----------
left, right = st.columns(2)

with left:
    st.subheader("Số khách theo nhóm")
    cnt = dff["Cluster"].value_counts().reset_index()
    cnt.columns = ["Cluster", "Customers"]
    fig1 = px.pie(cnt, names="Cluster", values="Customers", hole=0.55,
                  color="Cluster", color_discrete_map=color_map)
    fig1.update_traces(textinfo="percent", sort=False)
    fig1.update_layout(margin=dict(t=10, b=10, l=10, r=10), legend_title_text="Nhóm")
    st.plotly_chart(fig1, use_container_width=True)

with right:
    st.subheader("Doanh thu theo nhóm")
    revenue = rev_by.reset_index()
    revenue["Share"] = revenue["Monetary"] / total_rev
    fig2 = px.bar(revenue, x="Cluster", y="Monetary", color="Cluster",
                  color_discrete_map=color_map,
                  text=revenue["Share"].map("{:.0%}".format))
    fig2.update_traces(textposition="outside")
    fig2.update_layout(showlegend=False, xaxis_title="Nhóm",
                       yaxis_title="Doanh thu", margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig2, use_container_width=True)

# ---------- Phân tích RFM ----------
st.subheader("Mua gần đây và chi tiêu")
st.caption("Kích thước bong bóng là số lần mua. Trục chi tiêu dùng thang log, "
           "nên chỉ vẽ khách có chi tiêu lớn hơn 0 (tối đa 5.000 điểm ngẫu nhiên).")
plot_df = dff[dff["Monetary"] > 0]
plot_df = plot_df.sample(min(len(plot_df), 5000), random_state=42)
fig3 = px.scatter(plot_df, x="Recency", y="Monetary", color="Cluster",
                  size="Frequency", size_max=24, log_y=True, opacity=0.7,
                  color_discrete_map=color_map, hover_data=["Customer_ID"],
                  labels={"Recency": "Số ngày từ lần mua cuối", "Monetary": "Chi tiêu"})
fig3.update_layout(margin=dict(t=10, b=10, l=10, r=10), legend_title_text="Nhóm")
st.plotly_chart(fig3, use_container_width=True)

# ---------- Bảng đặc điểm ----------
st.subheader("Đặc điểm từng nhóm")
summary = (dff.groupby("Cluster")
           .agg(Customers=("Customer_ID", "count"),
                Recency=("Recency", "mean"),
                Frequency=("Frequency", "mean"),
                Monetary=("Monetary", "mean"),
                Revenue=("Monetary", "sum"))
           .reset_index())
summary["Share"] = summary["Revenue"] / total_rev * 100
summary = summary.round(1)

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Cluster": st.column_config.TextColumn("Nhóm"),
        "Customers": st.column_config.NumberColumn("Số khách", format="%d"),
        "Recency": st.column_config.NumberColumn("Recency TB (ngày)", format="%.1f"),
        "Frequency": st.column_config.NumberColumn("Frequency TB", format="%.1f"),
        "Monetary": st.column_config.NumberColumn("Monetary TB", format="%.0f"),
        "Revenue": st.column_config.NumberColumn("Doanh thu", format="%.0f"),
        "Share": st.column_config.ProgressColumn("Tỷ trọng doanh thu", format="%.1f%%",
                                                 min_value=0, max_value=100),
    },
)

st.download_button("Tải bảng tổng hợp (CSV)",
                   summary.to_csv(index=False).encode("utf-8"),
                   file_name="customer_segments_summary.csv", mime="text/csv")
