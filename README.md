# E-commerce Big Data: Phân khúc khách hàng, dự báo và thống kê doanh thu

Dự án xây dựng một quy trình xử lý dữ liệu lớn (Big Data) cho dữ liệu bán lẻ trực tuyến, chạy hoàn toàn bằng Docker trên cụm Hadoop (HDFS) và Spark. Kết quả được hiển thị trên dashboard Streamlit.

> **Sinh viên thực hiện:** [Họ tên] · **MSSV:** [...] · **Lớp:** [...] · **Giảng viên:** [...]

## 1. Mục tiêu

1. **Phân khúc khách hàng** bằng phân tích RFM (Recency, Frequency, Monetary) kết hợp thuật toán KMeans.
2. **Dự đoán doanh thu tương lai**, so sánh nhiều mô hình học máy (MLlib và scikit-learn).
3. **Thống kê doanh thu thực tế**: tính cả đơn hàng bị hủy/hoàn trả và các khoản chi phí đặc biệt của cửa hàng.

## 2. Dữ liệu

- Bộ dữ liệu **Online Retail II** (UCI / Kaggle), các giao dịch của một cửa hàng bán lẻ trực tuyến tại Anh, giai đoạn 2009-2011.
- File gốc đặt tại `data/online_retail_II.csv`.
- Nếu file quá nặng không kèm theo được, tải về từ Kaggle và đặt đúng tên file trên.

## 3. Kiến trúc

```
data/online_retail_II.csv
        │  make upload-data
        ▼
HDFS  /ecommerce/bronze/online_retail_II.csv        (dữ liệu thô)
        │  01_etl_cleaning
        ▼
HDFS  /ecommerce/silver/transactions_cleaned        (đã làm sạch)
        │  02_prepare_gold → 03_rfm_kmeans → 04_prepare_forecast → 05_data_summary
        ▼
HDFS  /ecommerce/gold/...                           (dữ liệu phân tích)
        │  06_calculate_actual_revenue
        ▼
HDFS  /ecommerce/results/actual_revenue/...         (kết quả thống kê)
        │
        ▼
Streamlit dashboard  (http://localhost:8501)
```

Dữ liệu đi qua ba tầng **bronze → silver → gold** (kiến trúc medallion). Mọi bước tính toán chạy bằng Spark, đọc và ghi trực tiếp trên HDFS.

### Công nghệ

| Thành phần | Phiên bản / ghi chú |
|---|---|
| Hadoop (HDFS, YARN) | `bde2020/hadoop-*:2.0.0-hadoop3.2.1-java8` |
| Spark | `bde2020/spark-*:3.0.0-hadoop3.2` (1 master, 1 worker) |
| PySpark, MLlib | trong container `etl` |
| scikit-learn | trong container `ml` (thí nghiệm mô hình) |
| Streamlit, Plotly | dashboard |
| Docker Compose | điều phối toàn bộ dịch vụ |

## 4. Cấu trúc thư mục

```
ecommerce-bigdata/
├── app.py                          # trang chính của dashboard
├── pages/                          # các trang phụ của dashboard (vd: 3_Revenue.py)
├── scripts/
│   ├── 01_etl_cleaning.py          # làm sạch dữ liệu
│   ├── 02_prepare_gold.py          # chuẩn bị dữ liệu gold
│   ├── 03_rfm_kmeans.py            # phân khúc khách hàng RFM + KMeans
│   ├── 04_prepare_forecast.py      # chuẩn bị dữ liệu dự báo
│   ├── 05_data_summary.py          # tổng hợp data_summary
│   ├── 06_calculate_actual_revenue.py  # doanh thu thực tế
│   ├── run_pipeline.sh             # chạy toàn bộ pipeline (01 → 06)
│   ├── run_experiments.sh          # chạy lại các thí nghiệm mô hình
│   └── experiments/
│       ├── mllib/                  # thí nghiệm với Spark MLlib
│       └── sklearn/                # thí nghiệm với scikit-learn
├── docker/
│   ├── docker-compose.yml          # định nghĩa toàn bộ dịch vụ
│   └── hadoop.env                  # cấu hình Hadoop
├── data/                           # dữ liệu gốc (CSV)
├── Dockerfile                      # image cho container etl
├── Dockerfile.ml                   # image cho container ml
├── Dockerfile.streamlit            # image cho dashboard
├── Makefile                        # các lệnh chạy nhanh
└── README.md
```

## 5. Yêu cầu hệ thống

- Linux (Ubuntu hoặc máy ảo Linux), đã cài **Docker** và **Docker Compose**.
- RAM tối thiểu **8 GB** (khuyến nghị 12 GB trở lên), ổ cứng trống khoảng **15 GB**.
- Có Internet ở lần chạy đầu để tải các image Hadoop và Spark.
- Cần cài `make` (`sudo apt install make`).

## 6. Hướng dẫn chạy

Chạy lần lượt từng bước trong thư mục dự án:

```bash
# 1. Cấp quyền chạy cho các script (chỉ cần làm một lần)
chmod +x scripts/run_pipeline.sh scripts/run_experiments.sh

# 2. Khởi động hệ thống (Hadoop, Spark, Streamlit, ...)
make up
# Chờ khoảng 1-2 phút, kiểm tra các container đều ở trạng thái Up:
docker ps

# 3. Đưa dữ liệu gốc lên HDFS
make upload-data

# 4. Chạy toàn bộ pipeline (làm sạch, RFM, dự báo, doanh thu)
make pipeline

# 5. Bật dashboard
make dashboard
```

Mở trình duyệt vào **http://localhost:8501** để xem kết quả.

Có thể chạy một lệnh duy nhất: `make all` (gồm `up`, `upload-data`, `pipeline`, `dashboard`).

### Các địa chỉ giao diện

| Dịch vụ | Địa chỉ |
|---|---|
| Dashboard Streamlit | http://localhost:8501 |
| HDFS NameNode | http://localhost:9870 |
| Spark Master | http://localhost:8080 |
| YARN ResourceManager | http://localhost:8088 |

Nếu chạy trong máy ảo và mở trình duyệt từ máy thật, thay `localhost` bằng địa chỉ IP của máy ảo.

## 7. Các lệnh `make`

| Lệnh | Chức năng |
|---|---|
| `make help` | Xem tất cả lệnh |
| `make up` / `make down` | Bật / tắt hệ thống (giữ dữ liệu) |
| `make build` | Build lại các image |
| `make ps` / `make logs` | Xem trạng thái / log container |
| `make upload-data` | Đưa file CSV gốc lên HDFS (tầng bronze) |
| `make etl` | Bước 01: làm sạch dữ liệu |
| `make rfm` | Bước 02 + 03: gold và RFM/KMeans |
| `make forecast` | Bước 04: chuẩn bị dữ liệu dự báo |
| `make summary` | Bước 05: tổng hợp data_summary |
| `make revenue` | Bước 06: tính doanh thu thực tế |
| `make pipeline` | Chạy toàn bộ các bước 01 → 06 |
| `make dashboard` | Bật dashboard Streamlit |
| `make experiments` | Chạy lại các thí nghiệm so sánh mô hình (lâu hơn pipeline chính) |
| `make experiments-mllib` / `make experiments-sklearn` | Chỉ chạy thí nghiệm MLlib / scikit-learn |
| `make clean` | Tắt và xóa container (giữ dữ liệu HDFS) |
| `make reset` | Tắt và **xóa toàn bộ dữ liệu HDFS** |

Chạy riêng một thí nghiệm: `make exp-mllib SCRIPT=03_random_forest.py` hoặc `make exp-sklearn SCRIPT=04_extra_trees.py`.

## 8. Pipeline xử lý

| Bước | Script | Việc làm | Đầu ra chính trên HDFS |
|---|---|---|---|
| 01 | `01_etl_cleaning.py` | Đọc dữ liệu thô, làm sạch | `/ecommerce/silver/transactions_cleaned` |
| 02 | `02_prepare_gold.py` | Chuẩn bị dữ liệu tầng gold | `/ecommerce/gold/...` |
| 03 | `03_rfm_kmeans.py` | Tính RFM, phân cụm KMeans | phân khúc khách hàng |
| 04 | `04_prepare_forecast.py` | Chuẩn bị dữ liệu cho mô hình dự báo | dữ liệu dự báo |
| 05 | `05_data_summary.py` | Tổng hợp dữ liệu | `/ecommerce/gold/data_summary` |
| 06 | `06_calculate_actual_revenue.py` | Tính doanh thu thực tế | `/ecommerce/results/actual_revenue/...` |

### Cách tính doanh thu thực tế (bước 06)

```
Doanh thu thực tế = Doanh thu bán hàng − Tiền hàng trả lại − Chi phí mã đặc biệt
```

- **Doanh thu bán hàng:** các dòng hóa đơn thường (không bắt đầu bằng "C"), `Quantity > 0`, `Price > 0`, không thuộc mã đặc biệt.
- **Tiền hàng trả lại:** các dòng hóa đơn hủy (bắt đầu bằng "C") có `Quantity < 0`, `Price > 0`, không thuộc mã đặc biệt; lấy giá trị tuyệt đối.
- **Chi phí mã đặc biệt:** các dòng thuộc StockCode `AMAZONFEE`, `BANK CHARGES`, `CRUK`, `POST`, `DOT`, `C2` (cả hóa đơn thường lẫn hóa đơn hủy), lấy giá trị tuyệt đối.
- Kết quả được tổng hợp theo **ngày, tháng, năm**; đơn hủy được trừ vào ngày phát sinh hóa đơn hủy.

Các thư mục kết quả trong `/ecommerce/results/actual_revenue/`: `sales_revenue_daily`, `returns_daily`, `target_expenses_daily`, `actual_revenue_daily`, `actual_revenue_monthly`, `actual_revenue_yearly`.

**Kết quả doanh thu theo năm** (từ lần chạy thực tế):

| Năm | Doanh thu bán hàng | Hàng trả lại | Chi phí mã đặc biệt | Doanh thu thực tế |
|---|---|---|---|---|
| 2009 | 798,118.53 | 19,558.08 | 22,014.53 | 756,545.92 |
| 2010 | 9,375,309.09 | 238,620.75 | 185,846.82 | 8,950,841.52 |
| 2011 | 9,472,190.18 | 458,353.30 | 262,088.22 | 8,751,748.66 |

Lưu ý: dữ liệu năm 2009 và 2011 không đủ cả năm nên không so sánh trực tiếp với 2010.

## 9. Dashboard

- **Trang chính** (`app.py`) và các trang trong `pages/`.
- **Trang Revenue** (`pages/3_Revenue.py`) gồm: thẻ tổng quan, biểu đồ waterfall từ doanh thu bán hàng đến doanh thu thực tế, và các tab theo ngày, tháng, năm, hàng trả lại, mã StockCode đặc biệt. Có bộ lọc thời gian và nút tải dữ liệu CSV.
- Dữ liệu được lưu đệm 5 phút. Nhấn **"Tải lại dữ liệu"** ở thanh bên để làm mới ngay sau khi chạy lại pipeline.

## 10. Thí nghiệm mô hình dự báo

Thư mục `scripts/experiments/` chứa các thí nghiệm so sánh mô hình dự báo doanh thu:

- `mllib/`: Linear Regression, Decision Tree, Random Forest (Spark MLlib).
- `sklearn/`: Linear Regression, Decision Tree, Random Forest, Extra Trees (scikit-learn).

Các thí nghiệm **không** nằm trong `make pipeline`. Chạy bằng `make experiments`; log từng thí nghiệm lưu ở `logs/experiments/`.

**Kết quả so sánh mô hình:**

| Mô hình | Thư viện | RMSE | MAE | R² |
|---|---|---|---|---|
| Linear Regression | [...] | [...] | [...] | [...] |
| Decision Tree | [...] | [...] | [...] | [...] |
| Random Forest | [...] | [...] | [...] | [...] |
| Extra Trees | scikit-learn | [...] | [...] | [...] |

**Mô hình được chọn:** [tên mô hình] vì [lý do].

## 11. Xử lý sự cố

| Triệu chứng | Cách xử lý |
|---|---|
| `Makefile: *** missing separator` | Dòng lệnh trong Makefile phải thụt bằng ký tự **Tab**, không phải dấu cách |
| `No such container: etl` | Chưa chạy `make up`, hoặc container chưa lên. Kiểm tra bằng `docker ps` |
| `Permission denied` khi chạy script | `chmod +x scripts/run_pipeline.sh scripts/run_experiments.sh` |
| Pipeline báo kết nối `spark-master:7077` bị từ chối | Spark master chưa lên xong, chờ 30 giây rồi chạy lại |
| `make upload-data` không tìm thấy file | Kiểm tra `data/online_retail_II.csv` có tồn tại: `ls data` |
| Dashboard không có dữ liệu mới | Nhấn "Tải lại dữ liệu" ở thanh bên (cache 5 phút) |
| Container thoát bất thường, treo | Thiếu RAM, tăng bộ nhớ cho máy ảo (khuyến nghị 12 GB) |
| Muốn chạy lại từ đầu | `make reset`, rồi `make up`, `make upload-data`, `make pipeline` |

## 12. Hạn chế và hướng phát triển

- Cụm Spark chỉ có một worker, phù hợp cho thử nghiệm và học tập, chưa tối ưu cho dữ liệu rất lớn.
- Các mã đặc biệt như `POST`, `DOT` hiện được trừ hoàn toàn vào doanh thu thực tế. Nếu chúng là khoản khách trả thêm (như phí vận chuyển) thì nên tách thành khoản thu riêng.
- Có thể thêm bước huấn luyện mô hình dự báo cuối cùng vào pipeline chính, thay vì chỉ nằm trong thí nghiệm.
