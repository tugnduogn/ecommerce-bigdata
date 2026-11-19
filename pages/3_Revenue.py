
import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

import streamlit as st

from pyspark.sql import SparkSession



# =====================================================

# 1. CẤU HÌNH TRANG & GIAO DIỆN

# =====================================================



st.set_page_config(

    page_title="Thống kê doanh thu thực tế",

    page_icon="📊",

    layout="wide",

)



RESULTS = "hdfs://namenode:9000/ecommerce/results/actual_revenue"



# Bảng màu thống nhất cho toàn bộ biểu đồ

COLORS = {

    "sales": "#2E86DE",     # doanh thu bán hàng

    "returns": "#EE5253",   # hàng trả lại

    "expense": "#FF9F43",   # chi phí mã đặc biệt

    "actual": "#10AC84",    # doanh thu thực tế

    "neutral": "#8395A7",

}



# Tên cột hiển thị cho người dùng

LABELS = {

    "Date": "Ngày",

    "YearMonth": "Tháng",

    "Year": "Năm",

    "Sales_Revenue": "Doanh thu bán hàng",

    "Return_Amount": "Tiền hàng trả lại",

    "Target_Expense": "Chi phí mã đặc biệt",

    "Actual_Revenue": "Doanh thu thực tế",

    "Return_Order_Count": "Số đơn trả lại",

    "StockCode": "StockCode",

    "Target_Amount": "Chi phí",

    "Transaction_Count": "Số giao dịch",

}



REVENUE_COLUMNS = ["Sales_Revenue", "Return_Amount", "Target_Expense", "Actual_Revenue"]



st.markdown(

    """

    <style>

        .block-container {padding-top: 2rem; padding-bottom: 2rem;}

        div[data-testid="stMetric"] {

            background: rgba(131, 149, 167, 0.10);

            border: 1px solid rgba(131, 149, 167, 0.25);

            border-radius: 12px;

            padding: 14px 18px;

        }

        div[data-testid="stMetricLabel"] p {font-size: 0.9rem; opacity: 0.8;}

        div[data-testid="stMetricValue"] {font-size: 1.6rem; font-weight: 700;}

        button[data-baseweb="tab"] {font-size: 1rem; font-weight: 600;}

    </style>

    """,

    unsafe_allow_html=True,

)



st.title("📊 Thống kê doanh thu thực tế")

st.caption(

    "Doanh thu thực tế = Doanh thu bán hàng − Tiền hàng trả lại − Chi phí mã đặc biệt"

)





# =====================================================

# 2. KẾT NỐI SPARK & ĐỌC DỮ LIỆU TỪ HDFS

# =====================================================



@st.cache_resource

def get_spark():

    return (

        SparkSession.builder

        .appName("Streamlit_Revenue_Statistics")

        .master("local[*]")

        .getOrCreate()

    )





def load_csv(folder: str) -> pd.DataFrame:

    """Đọc thư mục CSV trên HDFS thành pandas DataFrame.



    inferSchema=False: Spark đọc mọi cột dưới dạng chuỗi, tránh lỗi

    'Casting to unit-less dtype datetime64' khi toPandas().

    Việc ép kiểu ngày/số được thực hiện bằng pandas ở bên dưới.

    """

    spark = get_spark()

    df = (

        spark.read

        .option("header", True)

        .option("inferSchema", False)

        .csv(f"{RESULTS}/{folder}")

    )

    return df.toPandas()





def to_numeric(df: pd.DataFrame, columns: list) -> pd.DataFrame:

    for column in columns:

        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    return df





@st.cache_data(ttl=300)

def load_revenue_daily() -> pd.DataFrame:

    df = load_csv("actual_revenue_daily")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df = to_numeric(df, REVENUE_COLUMNS)

    return df.dropna(subset=["Date"]).sort_values("Date")





@st.cache_data(ttl=300)

def load_revenue_monthly() -> pd.DataFrame:

    df = load_csv("actual_revenue_monthly")

    df = to_numeric(df, REVENUE_COLUMNS)

    df["YearMonth"] = df["YearMonth"].astype(str)

    return df.sort_values("YearMonth")





@st.cache_data(ttl=300)

def load_revenue_yearly() -> pd.DataFrame:

    df = load_csv("actual_revenue_yearly")

    df = to_numeric(df, REVENUE_COLUMNS)

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    return df.dropna(subset=["Year"]).sort_values("Year")





@st.cache_data(ttl=300)

def load_returns() -> pd.DataFrame:

    df = load_csv("returns_daily")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df = to_numeric(df, ["Return_Order_Count", "Return_Amount"])

    return df.dropna(subset=["Date"]).sort_values("Date")





@st.cache_data(ttl=300)

def load_target_expenses() -> pd.DataFrame:

    df = load_csv("target_expenses_daily")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df = to_numeric(df, ["Target_Amount", "Transaction_Count"])

    df["StockCode"] = df["StockCode"].astype(str)

    return df.dropna(subset=["Date"]).sort_values("Date")





try:

    with st.spinner("Đang đọc dữ liệu từ HDFS..."):

        daily = load_revenue_daily()

        monthly = load_revenue_monthly()

        yearly = load_revenue_yearly()

        returns = load_returns()

        target_expenses = load_target_expenses()

except Exception as e:

    st.error("Chưa đọc được dữ liệu doanh thu từ HDFS.")

    st.exception(e)

    st.stop()



if daily.empty:

    st.warning("Không có dữ liệu doanh thu theo ngày.")

    st.stop()





# =====================================================

# 3. HÀM TIỆN ÍCH

# =====================================================



def fmt(value: float, decimals: int = 2) -> str:

    return f"{value:,.{decimals}f}"





def style_figure(fig: go.Figure, title: str, x_title: str = "", y_title: str = "") -> go.Figure:

    fig.update_layout(

        title=dict(text=title, font=dict(size=18)),

        xaxis_title=x_title,

        yaxis_title=y_title,

        template="plotly_white",

        hovermode="x unified",

        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),

        margin=dict(l=10, r=10, t=70, b=10),

        height=430,

    )

    fig.update_yaxes(tickformat=",.0f", gridcolor="rgba(131,149,167,0.2)")

    return fig





def show_table(df: pd.DataFrame, file_name: str) -> None:

    """Hiển thị bảng với tên cột tiếng Việt và nút tải CSV."""

    table = df.copy()

    if "Date" in table.columns:

        table["Date"] = table["Date"].dt.strftime("%Y-%m-%d")

    if "Year" in table.columns:

        table["Year"] = table["Year"].astype(int)



    numeric_cols = table.select_dtypes("number").columns.difference(["Year"])

    config = {

        col: st.column_config.NumberColumn(LABELS.get(col, col), format="%,.2f")

        for col in numeric_cols

    }

    config.update({

        col: st.column_config.TextColumn(LABELS.get(col, col))

        for col in table.columns.difference(numeric_cols)

    })



    st.dataframe(

        table,

        use_container_width=True,

        hide_index=True,

        column_config=config,

    )

    st.download_button(

        "⬇️ Tải CSV",

        data=table.rename(columns=LABELS).to_csv(index=False).encode("utf-8-sig"),

        file_name=file_name,

        mime="text/csv",

    )





def add_revenue_traces(fig: go.Figure, df: pd.DataFrame, x: str, kind: str) -> None:

    """Thêm 2 chuỗi: doanh thu bán hàng và doanh thu thực tế."""

    series = [

        ("Sales_Revenue", COLORS["sales"]),

        ("Actual_Revenue", COLORS["actual"]),

    ]

    for column, color in series:

        if kind == "bar":

            fig.add_trace(go.Bar(

                x=df[x], y=df[column], name=LABELS[column], marker_color=color,

            ))

        else:

            fig.add_trace(go.Scatter(

                x=df[x], y=df[column], name=LABELS[column],

                mode="lines", line=dict(color=color, width=2),

            ))





# =====================================================

# 4. BỘ LỌC THỜI GIAN (SIDEBAR)

# =====================================================



st.sidebar.header("🔎 Bộ lọc dữ liệu")



min_date = daily["Date"].min().date()

max_date = daily["Date"].max().date()



date_range = st.sidebar.date_input(

    "Khoảng thời gian",

    value=(min_date, max_date),

    min_value=min_date,

    max_value=max_date,

)



# Xử lý trường hợp người dùng mới chọn một ngày

if isinstance(date_range, (tuple, list)):

    if len(date_range) == 2:

        start_date, end_date = date_range

    elif len(date_range) == 1:

        start_date = end_date = date_range[0]

    else:

        start_date, end_date = min_date, max_date

else:

    start_date = end_date = date_range



start_ts = pd.Timestamp(start_date)

end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)



st.sidebar.caption(f"Dữ liệu có sẵn: {min_date} → {max_date}")

st.sidebar.caption("Dữ liệu được lưu đệm 5 phút.")

if st.sidebar.button("🔄 Tải lại dữ liệu"):

    st.cache_data.clear()

    st.rerun()





def filter_by_date(df: pd.DataFrame) -> pd.DataFrame:

    return df[(df["Date"] >= start_ts) & (df["Date"] <= end_ts)].copy()





daily_f = filter_by_date(daily)

returns_f = filter_by_date(returns)

target_f = filter_by_date(target_expenses)



if daily_f.empty:

    st.warning("Không có dữ liệu trong khoảng thời gian đã chọn.")

    st.stop()



# Tháng: lấy cả tháng chứa ngày bắt đầu (so sánh theo đầu tháng)

monthly = monthly.copy()

monthly["_month_date"] = pd.to_datetime(monthly["YearMonth"] + "-01", errors="coerce")

month_start = start_ts.to_period("M").to_timestamp()

monthly_f = monthly[

    (monthly["_month_date"] >= month_start) & (monthly["_month_date"] <= end_ts)

].drop(columns=["_month_date"])



yearly_f = yearly[

    (yearly["Year"] >= start_ts.year) & (yearly["Year"] <= end_ts.year)

].copy()





# =====================================================

# 5. TỔNG QUAN

# =====================================================



total_sales = daily_f["Sales_Revenue"].sum()

total_returns = daily_f["Return_Amount"].sum()

total_expenses = daily_f["Target_Expense"].sum()

total_actual = daily_f["Actual_Revenue"].sum()



return_rate = total_returns / total_sales if total_sales else 0

expense_rate = total_expenses / total_sales if total_sales else 0

actual_rate = total_actual / total_sales if total_sales else 0



c1, c2, c3, c4 = st.columns(4)

c1.metric("💰 Doanh thu bán hàng", fmt(total_sales))

c2.metric("↩️ Tiền hàng trả lại", fmt(total_returns),

          delta=f"{return_rate:.1%} doanh thu", delta_color="off")

c3.metric("🏷️ Chi phí mã đặc biệt", fmt(total_expenses),

          delta=f"{expense_rate:.1%} doanh thu", delta_color="off")

c4.metric("✅ Doanh thu thực tế", fmt(total_actual),

          delta=f"{actual_rate:.1%} doanh thu", delta_color="off")



st.write("")





# =====================================================

# 6. CÁC TAB NỘI DUNG

# =====================================================



tab_overview, tab_daily, tab_monthly, tab_yearly, tab_returns, tab_target = st.tabs([

    "📌 Tổng quan",

    "📅 Theo ngày",

    "🗓️ Theo tháng",

    "📆 Theo năm",

    "↩️ Hàng trả lại",

    "🏷️ Mã đặc biệt",

])





# ---------- Tab: Tổng quan ----------

with tab_overview:

    fig_waterfall = go.Figure(go.Waterfall(

        x=["Doanh thu bán hàng", "Tiền hàng trả lại", "Chi phí mã đặc biệt", "Doanh thu thực tế"],

        y=[total_sales, -total_returns, -total_expenses, total_actual],

        measure=["absolute", "relative", "relative", "total"],

        text=[fmt(v, 0) for v in [total_sales, -total_returns, -total_expenses, total_actual]],

        textposition="outside",

        increasing=dict(marker=dict(color=COLORS["sales"])),

        decreasing=dict(marker=dict(color=COLORS["returns"])),

        totals=dict(marker=dict(color=COLORS["actual"])),

        connector=dict(line=dict(color=COLORS["neutral"], dash="dot")),

    ))

    style_figure(fig_waterfall, "Từ doanh thu bán hàng đến doanh thu thực tế")

    fig_waterfall.update_layout(hovermode="closest", showlegend=False)

    st.plotly_chart(fig_waterfall, use_container_width=True)



    st.caption(

        f"Khoảng thời gian: {start_date} → {end_date} "

        f"({daily_f['Date'].nunique():,} ngày có dữ liệu)"

    )





# ---------- Tab: Theo ngày ----------

with tab_daily:

    show_ma = st.checkbox("Hiển thị đường trung bình 7 ngày", value=True)



    fig_daily = go.Figure()

    add_revenue_traces(fig_daily, daily_f, "Date", "line")



    if show_ma:

        ma = daily_f.set_index("Date")["Actual_Revenue"].rolling(7, min_periods=1).mean()

        fig_daily.add_trace(go.Scatter(

            x=ma.index, y=ma.values, name="Thực tế (TB 7 ngày)",

            mode="lines", line=dict(color=COLORS["expense"], width=2, dash="dash"),

        ))



    style_figure(fig_daily, "Doanh thu bán hàng và doanh thu thực tế theo ngày", "Ngày", "Doanh thu")

    st.plotly_chart(fig_daily, use_container_width=True)



    st.subheader("Chi tiết doanh thu theo ngày")

    show_table(

        daily_f[["Date"] + REVENUE_COLUMNS],

        "doanh_thu_theo_ngay.csv",

    )





# ---------- Tab: Theo tháng ----------

with tab_monthly:

    if monthly_f.empty:

        st.info("Không có dữ liệu doanh thu theo tháng trong khoảng thời gian này.")

    else:

        fig_monthly = go.Figure()

        add_revenue_traces(fig_monthly, monthly_f, "YearMonth", "bar")

        style_figure(fig_monthly, "Doanh thu bán hàng và doanh thu thực tế theo tháng", "Tháng", "Doanh thu")

        fig_monthly.update_layout(barmode="group")

        st.plotly_chart(fig_monthly, use_container_width=True)



        st.subheader("Chi tiết doanh thu theo tháng")

        show_table(

            monthly_f[["YearMonth"] + REVENUE_COLUMNS],

            "doanh_thu_theo_thang.csv",

        )





# ---------- Tab: Theo năm ----------

with tab_yearly:

    if yearly_f.empty:

        st.info("Không có dữ liệu doanh thu theo năm trong khoảng thời gian này.")

    else:

        yearly_plot = yearly_f.assign(Year=yearly_f["Year"].astype(int).astype(str))



        fig_yearly = go.Figure()

        add_revenue_traces(fig_yearly, yearly_plot, "Year", "bar")

        style_figure(fig_yearly, "Doanh thu bán hàng và doanh thu thực tế theo năm", "Năm", "Doanh thu")

        fig_yearly.update_layout(barmode="group")

        st.plotly_chart(fig_yearly, use_container_width=True)



        st.subheader("Chi tiết doanh thu theo năm")

        show_table(

            yearly_f[["Year"] + REVENUE_COLUMNS],

            "doanh_thu_theo_nam.csv",

        )





# ---------- Tab: Hàng trả lại ----------

with tab_returns:

    if returns_f.empty:

        st.info("Không có dữ liệu đơn hàng trả lại trong khoảng thời gian này.")

    else:

        m1, m2 = st.columns(2)

        m1.metric("Tổng số đơn hàng trả lại", fmt(returns_f["Return_Order_Count"].sum(), 0))

        m2.metric("Tổng tiền hàng trả lại", fmt(returns_f["Return_Amount"].sum()))



        fig_returns = go.Figure()

        fig_returns.add_trace(go.Bar(

            x=returns_f["Date"], y=returns_f["Return_Amount"],

            name="Tiền hàng trả lại", marker_color=COLORS["returns"],

        ))

        fig_returns.add_trace(go.Scatter(

            x=returns_f["Date"], y=returns_f["Return_Order_Count"],

            name="Số đơn trả lại", mode="lines",

            line=dict(color=COLORS["neutral"], width=2), yaxis="y2",

        ))

        style_figure(fig_returns, "Hàng trả lại theo ngày", "Ngày", "Số tiền")

        fig_returns.update_layout(

            yaxis2=dict(title="Số đơn", overlaying="y", side="right", showgrid=False),

        )

        st.plotly_chart(fig_returns, use_container_width=True)



        st.subheader("Chi tiết đơn hàng trả lại")

        show_table(returns_f, "don_hang_tra_lai.csv")





# ---------- Tab: Mã StockCode đặc biệt ----------

with tab_target:

    if target_f.empty:

        st.info("Không có chi phí mã StockCode đặc biệt trong khoảng thời gian này.")

    else:

        target_summary = (

            target_f.groupby("StockCode", as_index=False)

            .agg(

                Target_Amount=("Target_Amount", "sum"),

                Transaction_Count=("Transaction_Count", "sum"),

            )

            .sort_values("Target_Amount", ascending=False)

        )



        fig_target = px.bar(

            target_summary,

            x="StockCode",

            y="Target_Amount",

            text=target_summary["Target_Amount"].map(lambda v: f"{v:,.0f}"),

            color_discrete_sequence=[COLORS["expense"]],

        )

        fig_target.update_traces(textposition="outside")

        style_figure(fig_target, "Tổng chi phí theo từng StockCode", "StockCode", "Tổng chi phí")

        fig_target.update_layout(hovermode="closest")

        fig_target.update_xaxes(type="category")

        st.plotly_chart(fig_target, use_container_width=True)



        left, right = st.columns([1, 2])

        with left:

            st.subheader("Tổng hợp theo StockCode")

            show_table(target_summary, "chi_phi_theo_stockcode.csv")

        with right:

            st.subheader("Chi tiết theo ngày và StockCode")

            show_table(target_f, "chi_phi_theo_ngay_stockcode.csv")





# =====================================================

# 7. CHÂN TRANG

# =====================================================



st.markdown("---")

st.caption(

    "Nguồn dữ liệu: HDFS /ecommerce/results/actual_revenue · "

    "Bộ nhớ đệm 5 phút · Nhấn “Tải lại dữ liệu” ở thanh bên để làm mới."

)
