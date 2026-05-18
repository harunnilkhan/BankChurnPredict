# Model Card - BankChurnPredict

## Model Details

| Field | Value |
| --- | --- |
| Model name | BankChurnPredict |
| Version | v1.0.0 |
| Problem type | Binary classification |
| Framework | scikit-learn |
| Runtime | Python 3.11 |
| Best local model | RandomForestClassifier |

## Intended Use

The model predicts whether a bank customer is likely to churn. It is designed for portfolio demonstration, backend integration practice, and MLOps fundamentals. It should not be used for production customer decisions without stronger validation, monitoring, and governance.

## Dataset

| Field | Value |
| --- | --- |
| Dataset | [Bank Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/saurabhbadole/bank-customer-churn-prediction-dataset) |
| Full data path | `data/raw/churn.csv` |
| Sample data path | `data/sample/sample_churn.csv` |
| Target | `Exited` |
| Target meaning | `0 = stayed`, `1 = churned` |
| Approximate class balance | Imbalanced, majority non-churn |

## Input Features

| Feature | Type | Description |
| --- | --- | --- |
| `CreditScore` | Numerical | Customer credit score |
| `Geography` | Categorical | Customer country |
| `Gender` | Categorical | Customer gender |
| `Age` | Numerical | Customer age |
| `Tenure` | Numerical | Years as customer |
| `Balance` | Numerical | Account balance |
| `NumOfProducts` | Numerical | Number of bank products |
| `HasCrCard` | Binary | Whether the customer has a credit card |
| `IsActiveMember` | Binary | Whether the customer is active |
| `EstimatedSalary` | Numerical | Estimated annual salary |

## Training Method

1. Identifier columns are dropped: `RowNumber`, `CustomerId`, `Surname`.
2. Numerical values are imputed with median and scaled with `StandardScaler`.
3. Categorical values are imputed with most frequent value and encoded with `OneHotEncoder`.
4. Data is split into train, validation, and test sets with stratification.
5. Logistic Regression, Random Forest (`class_weight="balanced"`), and Gradient Boosting are trained.
6. The best model is selected by F1-score (balances precision and recall).
7. An optimal classification threshold is computed using the precision-recall curve.
8. Missing value imputation is performed only on the training set to prevent data leakage.

## Evaluation Metrics

Current local metrics from `models/model_metadata.json`:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.848 |
| Precision | 0.6359 |
| Recall | 0.5921 |
| F1-score | 0.6132 |
| ROC-AUC | 0.8613 |
| Optimal Threshold | 0.5217 |

Confusion matrix:

```text
[[1455, 138],
 [166, 241]]
```

## Limitations

- The dataset is imbalanced, so recall for churned customers is lower than overall accuracy.
- The model is trained on a static dataset and does not automatically adapt to drift.
- The feature set is small compared with real churn systems.
- Geography and gender may introduce fairness concerns and should be evaluated before real use.
- The sample dataset exists for CI and demos, not for final model quality.

## Ethical Considerations

- Predictions should support human decision-making, not replace it.
- Customers should not be treated adversely based only on an automated churn score.
- Fairness checks should be added before production use, especially for geography and gender.
- Input data should be handled under appropriate privacy and retention policies.

## Recommended Improvements

- Add class weighting or threshold tuning to improve churn recall.
- Track model versions and experiments with MLflow.
- Add monitoring for drift, latency, and prediction distributions.
- Add production-grade database support with PostgreSQL.
- Add retraining and model promotion workflow.
