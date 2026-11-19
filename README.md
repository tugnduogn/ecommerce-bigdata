# Ecommerce Big Data: Phân khúc khách hàng và dự báo doanh thu

Dự án xử lý dữ liệu bán lẻ trực tuyến theo kiến trúc big data: lưu trữ trên **HDFS** theo mô hình **Medallion (Bronze, Silver, Gold)**, xử lý bằng **Apache Spark**, phân tích bằng **Spark MLlib / scikit-learn** và hiển thị bằng **Streamlit**. Toàn bộ hạ tầng chạy bằng Docker Compose và được điều khiển bằng `make`.

## 1. Mục tiêu

- Phân khúc khách hàng bằng phân tích **RFM** (Recency, Frequency, Monetary) kết hợp **KMeans**.
- Dự đoán doanh thu tương lai, so sánh nhiều mô hình (Spark MLlib và scikit-learn).
- Thống kê doanh thu thực tế, tính cả đơn hàng bị hủy/hoàn trả và các khoản chi phí đặc biệt của cửa hàng.

**Dữ liệu:** bộ *Online Retail II* (UCI / Kaggle), giao dịch của một cửa hàng bán lẻ trực tuyến tại Anh, giai đoạn 2009–2011. File gốc nằm ở `data/online_retail_II.csv`.

## 2. Công nghệ

| Nhóm | Công cụ | Mục đích |
|---|---|---|
| Hạ tầng | Docker Compose | Dựng và quản lý HDFS, Spark, các dịch vụ |
| Hạ tầng | Dockerfile | Đóng gói script xử lý dữ liệu, ML và dashboard |
| Lưu trữ | HDFS | Lưu dữ liệu theo kiến trúc Medallion |
| Xử lý | Apache Spark (PySpark) | Làm sạch dữ liệu, tính RFM, tổng hợp doanh thu |
| Học máy | Spark MLlib, scikit-learn | KMeans, dự báo doanh thu (Linear Regression, Random Forest, Decision Tree...) |
| Trực quan | Streamlit | Dashboard hiển thị kết quả |
| Ngôn ngữ | Python | Toàn bộ pipeline |
| Mã nguồn | Git / GitHub | Quản lý code |

## 3. Luồng xử lý

```
data/online_retail_II.csv
        │  make upload-data
        ▼
HDFS  /ecommerce/bronze   (dữ liệu thô, giữ nguyên CSV)
        │  01_etl_cleaning.py
        ▼
HDFS  /ecommerce/silver   (dữ liệu đã làm sạch, có cột Revenue)
        │  02 → 06
        ▼
HDFS  /ecommerce/gold     (RFM, customer_segments, dữ liệu dự báo, data_summary, doanh thu thực tế)
        │
        ▼
Streamlit dashboard  →  http://localhost:8501
```

| Bước | Script | Chức năng |
|---|---|---|
| 01 | `01_etl_cleaning.py` | Làm sạch dữ liệu, thêm cột Revenue, ghi vào Silver |
| 02 | `02_prepare_gold.py` | Chuẩn bị dữ liệu tầng Gold |
| 03 | `03_rfm_kmeans.py` | Tính RFM, phân cụm KMeans, ghi `gold/customer_segments` |
| 04 | `04_prepare_forecast.py` | Chuẩn bị dữ liệu cho dự báo |
| 05 | `05_data_summary.py` | Tổng hợp `gold/data_summary` |
| 06 | `06_calculate_actual_revenue.py` | Tính doanh thu thực tế (đọc từ `gold/data_summary`) |

> Mỗi bước đọc kết quả của bước trước, nên **phải chạy đúng thứ tự**.

## 4. Cấu trúc thư mục

```
ecommerce-bigdata/
├── app.py                          # Trang chính của dashboard
├── pages/                          # Các trang phụ của dashboard
├── scripts/
│   ├── 01_etl_cleaning.py
│   ├── 02_prepare_gold.py
│   ├── 03_rfm_kmeans.py
│   ├── 04_prepare_forecast.py
│   ├── 05_data_summary.py
│   ├── 06_calculate_actual_revenue.py
│   ├── run_pipeline.sh             # Chạy pipeline (01 → 06)
│   ├── run_experiments.sh          # Chạy các thí nghiệm mô hình
│   └── experiments/
│       ├── mllib/                  # Thí nghiệm Spark MLlib
│       └── sklearn/                # Thí nghiệm scikit-learn
├── docker/
│   ├── docker-compose.yml          # Định nghĩa toàn bộ dịch vụ
│   └── hadoop.env                  # Cấu hình Hadoop
├── data/                           # Dữ liệu gốc (CSV)
├── Dockerfile                      # Image cho container etl
├── Dockerfile.ml                   # Image cho container ml
├── Dockerfile.streamlit            # Image cho dashboard
├── Makefile                        # Các lệnh chạy nhanh
└── README.md
```

## 5. Thành phần hệ thống (Docker)

| Container | Vai trò |
|---|---|
| namenode, datanode | Lưu trữ HDFS |
| resourcemanager, nodemanager | Quản lý tài nguyên YARN |
| spark-master, spark-worker | Cụm Spark |
| etl | Chạy các script pipeline (01 → 06) và thí nghiệm MLlib |
| ml | Chạy thí nghiệm scikit-learn |
| streamlit | Dashboard |

## 6. Yêu cầu hệ thống

- Docker và Docker Compose (bản có lệnh `docker compose`), `make`
- RAM tối thiểu **8GB dành cho Docker** (máy nên có 12–16GB)
- Ổ cứng trống khoảng 10GB, có mạng ở lần chạy đầu để tải image
- Cổng `8501` (dashboard) không bị chiếm

## 7. Cách chạy nhanh (một lệnh)

```bash
git clone https://github.com/tugnduogn/ecommerce-bigdata.git
cd ecommerce-bigdata
make all
```

`make all` thực hiện lần lượt: bật hạ tầng, đưa dữ liệu thô lên HDFS, chạy pipeline 01 → 06, bật dashboard. Lần đầu sẽ lâu (vài phút đến vài chục phút) vì phải build image và tải image Hadoop/Spark.

Khi chạy xong, mở trình duyệt: **http://localhost:8501**

> Lưu ý: `make up` **chỉ bật container**, không đưa dữ liệu lên HDFS và không chạy pipeline. Muốn có kết quả trên dashboard, dùng `make all` hoặc chạy từng bước như mục 8.

## 8. Chạy thủ công từng bước

Dùng khi muốn quan sát kết quả của từng tầng dữ liệu.

```bash
make up              # 1. Bật hạ tầng (chờ 1-2 phút cho HDFS sẵn sàng)
make upload-data     # 2. Đưa CSV thô lên HDFS  → /ecommerce/bronze
make etl             # 3. Bước 01: làm sạch      → /ecommerce/silver
make rfm             # 4. Bước 02 + 03: Gold, RFM, KMeans
make forecast        # 5. Bước 04: chuẩn bị dữ liệu dự báo
make summary         # 6. Bước 05: tổng hợp data_summary
make revenue         # 7. Bước 06: doanh thu thực tế
make dashboard       # 8. Bật Streamlit → http://localhost:8501
```

Kiểm tra kết quả sau mỗi bước:

```bash
docker exec namenode hdfs dfs -ls -R /ecommerce
```

Chạy lại toàn bộ bước 01 → 06 một lần: `make pipeline`.

## 9. Thí nghiệm mô hình dự báo

Các mô hình dự báo **không nằm trong pipeline chính**. Chạy sau khi đã hoàn tất `make forecast`.

| Lệnh | Việc làm |
|---|---|
| `make experiments` | Chạy tất cả thí nghiệm (MLlib và scikit-learn) |
| `make experiments-mllib` | Chỉ chạy nhóm Spark MLlib |
| `make experiments-sklearn` | Chỉ chạy nhóm scikit-learn |
| `make exp-mllib SCRIPT=<file>.py` | Chạy một script MLlib |
| `make exp-sklearn SCRIPT=<file>.py` | Chạy một script scikit-learn |

Xem tên script có sẵn: `ls scripts/experiments/mllib/ scripts/experiments/sklearn/`.
Kết quả scikit-learn: `scripts/experiments/sklearn/results_sklearn.csv`.

## 10. Các lệnh quản lý

| Lệnh | Tác dụng |
|---|---|
| `make help` | Liệt kê các lệnh |
| `make ps` | Xem trạng thái container |
| `make logs` | Xem log các container |
| `make build` | Build lại image etl, ml, streamlit |
| `make down` / `make clean` | Tắt và xóa container, **giữ** dữ liệu HDFS |
| `make reset` | Tắt và **xóa toàn bộ dữ liệu HDFS** (có 5 giây để hủy bằng Ctrl+C) |

Chạy lại từ đầu mà không dựng lại container: xóa dữ liệu HDFS rồi chạy lại từ mục 8.

```bash
docker exec namenode hdfs dfs -rm -r -f /ecommerce
```

## 11. Chạy trên Windows

Dùng **Docker Desktop + WSL2 (Ubuntu)** để chạy `make` như trên Linux.

1. Bật ảo hóa (VT-x / SVM) trong BIOS nếu chưa bật.
2. Mở PowerShell bằng quyền Administrator: `wsl --install -d Ubuntu`, khởi động lại máy và tạo user cho Ubuntu.
3. Cài **Docker Desktop** (backend WSL2), vào Settings, Resources, WSL Integration, bật **Ubuntu**, bấm Apply & Restart.
4. Cấp RAM cho WSL: tạo file `C:\Users\<TenUser>\.wslconfig`:
   ```
   [wsl2]
   memory=8GB
   processors=4
   ```
   rồi chạy `wsl --shutdown` và mở lại Docker Desktop.
5. Mở ứng dụng **Ubuntu** (không dùng PowerShell) và chạy:
   ```bash
   sudo apt update && sudo apt install -y make git
   cd ~
   git clone https://github.com/tugnduogn/ecommerce-bigdata.git
   cd ecommerce-bigdata
   make all
   ```

> Hãy đặt dự án trong thư mục home của Ubuntu (`~`), **không** đặt trong `/mnt/c/...` vì chậm và dễ lỗi quyền file, ký tự xuống dòng.

