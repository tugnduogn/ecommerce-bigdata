.PHONY: upload-data help up down restart build ps logs wait etl rfm forecast summary revenue pipeline all dashboard exp-mllib exp-sklearn clean reset experiments experiments-mllib experiments-sklearn

COMPOSE = cd docker && docker compose

help:
	@echo "make up         - bat ha tang (Hadoop, Spark, Streamlit, etl, ml)"
	@echo "make down       - tat ha tang (giu du lieu)"
	@echo "make build      - build lai image etl, ml, streamlit"
	@echo "make ps         - xem trang thai container"
	@echo "make logs       - xem log cac container"
	@echo "make etl        - buoc 01: lam sach du lieu"
	@echo "make rfm        - buoc 02 + 03: gold va RFM/KMeans"
	@echo "make forecast   - buoc 04: chuan bi du lieu du bao"
	@echo "make summary    - buoc 05: tong hop data_summary"
	@echo "make revenue    - buoc 06: tinh doanh thu thuc te"
	@echo "make pipeline   - chay toan bo buoc 01 -> 06"
	@echo "make all        - up + pipeline + dashboard"
	@echo "make dashboard  - bat Streamlit (http://localhost:8501)"
	@echo "make clean      - tat va xoa container (giu du lieu HDFS)"
	@echo "make reset      - tat va XOA TOAN BO du lieu HDFS"

# ---------- Ha tang ----------
up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

restart: down up

build:
	$(COMPOSE) build etl ml streamlit

ps:
	$(COMPOSE) ps

logs:
	$(COMPOSE) logs -f --tail=100

# ---------- Pipeline (chay trong container etl, qua scripts/run_pipeline.sh) ----------
etl:
	./scripts/run_pipeline.sh 01_etl

rfm:
	./scripts/run_pipeline.sh 02_prepare_gold
	./scripts/run_pipeline.sh 03_rfm

forecast:
	./scripts/run_pipeline.sh 04_prepare_forecast

summary:
	./scripts/run_pipeline.sh 05_data_summary

revenue:
	./scripts/run_pipeline.sh 06_calculate_actual_revenue

pipeline:
	./scripts/run_pipeline.sh

# ---------- Dashboard ----------
dashboard:
	$(COMPOSE) up -d streamlit
	@echo "Dashboard: http://localhost:8501"

all: up upload-data pipeline dashboard

# ---------- Thi nghiem mo hinh (KHONG nam trong pipeline chinh) ----------
# Vi du: make exp-mllib SCRIPT=03_random_forest.py
exp-mllib:
	@test -n "$(SCRIPT)" || (echo "Thieu SCRIPT=ten_file.py" && exit 1)
	docker exec etl python3 /app/scripts/experiments/mllib/$(SCRIPT)

# Vi du: make exp-sklearn SCRIPT=04_extra_trees.py
exp-sklearn:
	@test -n "$(SCRIPT)" || (echo "Thieu SCRIPT=ten_file.py" && exit 1)
	docker exec ml python3 /app/$(SCRIPT)

# ---------- Don dep ----------
clean:
	$(COMPOSE) down

reset:
	@echo "CANH BAO: lenh nay xoa toan bo du lieu HDFS. Bam Ctrl+C trong 5 giay de huy."
	@sleep 5
	$(COMPOSE) down -v

experiments:
	./scripts/run_experiments.sh

experiments-mllib:
	./scripts/run_experiments.sh mllib

experiments-sklearn:
	./scripts/run_experiments.sh sklearn

RAW_FILE = data/online_retail_II.csv

upload-data:
	@test -f $(RAW_FILE) || (echo "Khong thay $(RAW_FILE). Kiem tra bang: ls data" && exit 1)
	@until docker exec namenode hdfs dfs -ls / >/dev/null 2>&1; do echo "Cho HDFS khoi dong..."; sleep 5; done
	docker exec namenode hdfs dfsadmin -safemode wait
	docker cp $(RAW_FILE) namenode:/tmp/online_retail_II.csv
	docker exec namenode hdfs dfs -mkdir -p /ecommerce/bronze
	docker exec namenode hdfs dfs -put -f /tmp/online_retail_II.csv /ecommerce/bronze/online_retail_II.csv
	docker exec namenode rm -f /tmp/online_retail_II.csv
	docker exec namenode hdfs dfs -ls /ecommerce/bronze
