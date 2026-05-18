# BankChurnPredict

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39-ff4b4b)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)
![Docker](https://img.shields.io/badge/docker-ready-blue)
![Tests](https://img.shields.io/badge/tests-pytest-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Production-style machine learning deployment platform with FastAPI, scikit-learn, Docker, automated tests, model evaluation, and SQLite prediction logging.

## Overview

BankChurnPredict demonstrates the full path from a trained churn model to a deployable API service. The project trains and compares multiple scikit-learn classifiers, saves the best model and preprocessing pipeline, serves predictions through FastAPI, validates inputs with Pydantic, logs every prediction to SQLite, and includes Docker plus GitHub Actions CI support.

The current use case is bank customer churn prediction. The model predicts whether a customer is likely to leave the bank based on account and demographic features. A Streamlit dashboard is included for interactive demos on top of the FastAPI backend.

## Features

- CSV data loading, cleaning, missing-value handling, encoding, and scaling
- Model comparison across Logistic Regression, Random Forest, and Gradient Boosting
- Automatic best-model selection using F1-score, with ROC-AUC as fallback
- Saved model, preprocessor, and metadata artifacts
- Reusable inference module independent from FastAPI
- FastAPI endpoints for health, model info, single prediction, batch prediction, and prediction history
- Pydantic request and response schemas with field constraints
- SQLite prediction logging with request payload, result, probability, model version, and timestamp
- Streamlit dashboard for predictions, model metadata, and recent history
- Automated pytest suite and GitHub Actions workflow
- Dockerfile and Docker Compose support
- Professional project documentation and model card

## Tech Stack

Python 3.11, FastAPI, Uvicorn, Streamlit, Pydantic, pandas, NumPy, scikit-learn, joblib, SQLAlchemy, SQLite, pytest, httpx, requests, Docker, Docker Compose, GitHub Actions.

## Architecture

```text
Client
  |
  v
FastAPI app (app/main.py) and Streamlit dashboard (dashboard/app.py)
  |
  +-- Health router: GET /health
  +-- Model router: GET /model/info
  +-- Prediction router: POST /predict, POST /predict/batch, GET /predictions/history
  |
  v
Service layer
  |
  +-- model_loader.py loads cached model artifacts
  +-- prediction_service.py preprocesses input and calls the model
  +-- history_service.py writes and reads SQLite prediction logs
  |
  v
Models directory + SQLite database
```

Training is intentionally separate from serving:

```text
data/raw/churn.csv or data/sample/sample_churn.csv
  -> ml/preprocess.py
  -> ml/train.py
  -> models/best_model.joblib
  -> models/preprocessor.joblib
  -> models/model_metadata.json
```

More detail is available in [docs/architecture.md](docs/architecture.md).

## Dataset

The project uses the [Bank Customer Churn Prediction Dataset](https://www.kaggle.com/datasets/saurabhbadole/bank-customer-churn-prediction-dataset) from Kaggle, with the target column `Exited`.

Input features:

- `CreditScore`
- `Geography`
- `Gender`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`

The full local dataset is expected at `data/raw/churn.csv`. A committed sample dataset is available at `data/sample/sample_churn.csv` so training and CI can still run from a clean clone.

The training command also generates an inspectable processed sample at `data/processed/processed_churn_sample.csv`. That generated CSV is ignored by git because it can be recreated at any time.

## Model Training

Run the training pipeline:

```bash
python -m ml.train
```

The pipeline:

1. Loads the raw dataset, or falls back to the sample dataset.
2. Drops identifier columns.
3. Imputes missing values.
4. Scales numerical features and one-hot encodes categorical features.
5. Trains Logistic Regression, Random Forest, and Gradient Boosting.
6. Selects the best model by F1-score to balance precision and recall.
7. Saves artifacts to `models/`.

Current local training result:

```json
{
  "best_model": "RandomForestClassifier",
  "accuracy": 0.842,
  "precision": 0.6066,
  "recall": 0.6364,
  "f1_score": 0.6211,
  "roc_auc": 0.8584,
  "optimal_threshold": 0.4774,
  "confusion_matrix": [[1425, 168], [148, 259]]
}
```

## API Endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Service health check |
| GET | `/model/info` | Current model metadata and metrics |
| POST | `/predict` | Single churn prediction |
| POST | `/predict/batch` | Batch churn prediction |
| GET | `/predictions/history` | Prediction log history |

Interactive docs are available at `http://localhost:8000/docs` after starting the API.

The Streamlit dashboard is available at `http://localhost:8501` after starting the dashboard process.

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m ml.train
uvicorn app.main:app --reload
```

On macOS/Linux, activate the environment with:

```bash
source venv/bin/activate
```

PowerShell can also run everything without activating the virtual environment:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m ml.train
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

## Streamlit Dashboard

Start the FastAPI service first:

```bash
uvicorn app.main:app --reload
```

Then open a second terminal and run:

```bash
streamlit run dashboard/app.py
```

On Windows PowerShell without activating the virtual environment:

```powershell
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload
.\venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

Open the dashboard at `http://localhost:8501`.

## Example Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 619,
    "Geography": "France",
    "Gender": "Female",
    "Age": 42,
    "Tenure": 2,
    "Balance": 0.0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 101348.88
  }'
```

Example response:

```json
{
  "prediction": 0,
  "label": "No Churn Risk",
  "probability": 0.3061,
  "model_version": "v1.0.0"
}
```

More examples are available in [docs/api_examples.md](docs/api_examples.md).

## Docker Usage

The Docker image trains the model during build. If `data/raw/churn.csv` is not present, it falls back to the committed sample dataset.

Build and run with Docker:

```bash
docker build -t bank-churn-predict .
docker run -p 8000:8000 bank-churn-predict
```

Run with Docker Compose:

```bash
docker compose up --build
```

The compose setup starts both the API at `http://localhost:8000` and the Streamlit dashboard at `http://localhost:8501`. SQLite logs are persisted in a named volume mounted at `/app/data`.

## Testing

```bash
pytest
```

If `pytest` is not on your PATH, run:

```bash
python -m pytest
```

The tests cover preprocessing, artifact metadata, health/model endpoints, prediction validation, batch predictions, and prediction history.

For optional local linting and formatting checks:

```bash
pip install -r requirements-dev.txt
ruff check .
black --check .
```

## Project Structure

```text
app/                 FastAPI application, schemas, services, and database code
dashboard/           Streamlit dashboard for interactive demos
ml/                  Training, preprocessing, evaluation, registry, and inference code
data/raw/            Full raw dataset location
data/sample/         Small committed sample dataset
models/              Generated model artifacts
tests/               Automated pytest suite
docs/                Architecture, API examples, and model card
.github/workflows/  CI workflow
```

## Future Improvements

- Add PostgreSQL as an optional runtime database
- Add MLflow experiment tracking
- Add model version promotion rules
- Add API key authentication
- Add monitoring metrics for prediction volume and latency
- Deploy to Render, Railway, Fly.io, or a cloud container service

## License

MIT. See [LICENSE](LICENSE).
