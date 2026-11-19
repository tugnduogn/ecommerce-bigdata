FROM bde2020/spark-submit:3.0.0-hadoop3.2

# Cài numpy bằng apk
RUN apk add --no-cache py3-numpy

# Cài pyspark
RUN pip3 install pyspark==3.0.0

WORKDIR /app

# File ETL ở cấp ngoài của scripts/
COPY scripts/*.py /app/scripts/

# Chỉ chép phần MLlib
COPY scripts/experiments/mllib/ /app/scripts/experiments/mllib/

RUN chmod +x /app/scripts/*.py && \
    chmod +x /app/scripts/experiments/mllib/*.py

CMD ["/app/scripts/01_etl_cleaning.py"]
