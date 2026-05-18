# BankChurnPredict - API Examples

All examples assume the API is running at `http://localhost:8000`.

## Health Check

```bash
curl -X GET http://localhost:8000/health
```

```json
{
  "status": "ok",
  "service": "BankChurnPredict API"
}
```

## Model Information

```bash
curl -X GET http://localhost:8000/model/info
```

```json
{
  "project_name": "BankChurnPredict",
  "model_version": "v1.0.0",
  "problem_type": "binary_classification",
  "target_column": "Exited",
  "best_model": "RandomForestClassifier",
  "metrics": {
    "accuracy": 0.8675,
    "precision": 0.8318,
    "recall": 0.4373,
    "f1_score": 0.5733,
    "roc_auc": 0.8612,
    "confusion_matrix": [[1557, 36], [229, 178]]
  },
  "created_at": "2026-05-13 16:14:28"
}
```

## Single Prediction

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

```json
{
  "prediction": 0,
  "label": "No Churn Risk",
  "probability": 0.3061,
  "model_version": "v1.0.0"
}
```

## Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
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
      },
      {
        "CreditScore": 804,
        "Geography": "Spain",
        "Gender": "Male",
        "Age": 33,
        "Tenure": 7,
        "Balance": 76548.6,
        "NumOfProducts": 1,
        "HasCrCard": 0,
        "IsActiveMember": 1,
        "EstimatedSalary": 98453.45
      }
    ]
  }'
```

```json
{
  "results": [
    {
      "prediction": 0,
      "label": "No Churn Risk",
      "probability": 0.3061,
      "model_version": "v1.0.0"
    },
    {
      "prediction": 0,
      "label": "No Churn Risk",
      "probability": 0.0814,
      "model_version": "v1.0.0"
    }
  ],
  "count": 2
}
```

## Prediction History

```bash
curl -X GET http://localhost:8000/predictions/history
```

With pagination:

```bash
curl -X GET "http://localhost:8000/predictions/history?limit=5&offset=0"
```

```json
{
  "items": [
    {
      "id": 1,
      "input_data": {
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
      },
      "prediction": 0,
      "label": "No Churn Risk",
      "probability": 0.3061,
      "model_version": "v1.0.0",
      "created_at": "2026-05-13T15:53:59.180467"
    }
  ],
  "total": 1,
  "limit": 20,
  "offset": 0
}
```

## Interactive Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
