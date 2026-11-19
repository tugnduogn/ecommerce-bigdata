FROM python:3.9-slim

WORKDIR /app

# Cài đặt thư viện Machine Learning + đọc HDFS
RUN pip install --no-cache-dir pandas scikit-learn pyarrow hdfs

# Sao chép các file thực nghiệm
COPY scripts/experiments/sklearn/ /app/

CMD ["python", "01_linear_regression.py"]
