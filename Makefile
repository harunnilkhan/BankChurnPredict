.PHONY: install install-dev train run dashboard test docker-build docker-run docker-up lint format clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

train:
	python -m ml.train

run:
	uvicorn app.main:app --reload

dashboard:
	streamlit run dashboard/app.py

test:
	pytest

docker-build:
	docker build -t bank-churn-predict .

docker-run:
	docker run -p 8000:8000 bank-churn-predict

docker-up:
	docker compose up --build

lint:
	ruff check .
	black --check .

format:
	black .
	ruff check --fix .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f bankchurnpredict.db
