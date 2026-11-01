FROM bde2020/spark-submit:3.0.0-hadoop3.2

# Cài pyspark
RUN pip3 install pyspark==3.0.0

COPY scripts/01_etl_cleaning.py /app/01_etl_cleaning.py

WORKDIR /app

RUN chmod +x /app/01_etl_cleaning.py

CMD ["/app/01_etl_cleaning.py"]
