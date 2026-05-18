"""Streamlit dashboard for BankChurnPredict."""

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st


DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
API_BASE_URL = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


def get_json(path: str, timeout: int = 5) -> dict[str, Any]:
    """Fetch JSON from the FastAPI service."""
    response = requests.get(f"{API_BASE_URL}{path}", timeout=timeout)
    response.raise_for_status()
    return response.json()


def post_json(path: str, payload: dict[str, Any], timeout: int = 10) -> dict[str, Any]:
    """Post JSON to the FastAPI service."""
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=15)
def load_model_info() -> dict[str, Any] | None:
    """Load model metadata for dashboard display."""
    try:
        return get_json("/model/info")
    except requests.RequestException:
        return None


@st.cache_data(ttl=10)
def load_prediction_history(limit: int = 25) -> dict[str, Any] | None:
    """Load recent prediction history."""
    try:
        return get_json(f"/predictions/history?limit={limit}&offset=0")
    except requests.RequestException:
        return None


def build_prediction_payload() -> dict[str, Any]:
    """Render the prediction form and return its payload."""
    with st.form("prediction-form"):
        left, right = st.columns(2)

        with left:
            credit_score = st.number_input(
                "Credit score",
                min_value=300,
                max_value=900,
                value=619,
                step=1,
            )
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.number_input("Age", min_value=18, max_value=100, value=42, step=1)
            tenure = st.number_input(
                "Tenure",
                min_value=0,
                max_value=10,
                value=2,
                step=1,
            )

        with right:
            balance = st.number_input("Balance", min_value=0.0, value=0.0, step=100.0)
            num_products = st.number_input(
                "Number of products",
                min_value=1,
                max_value=4,
                value=1,
                step=1,
            )
            has_card = st.selectbox("Has credit card", [1, 0], format_func=yes_no)
            active_member = st.selectbox(
                "Is active member",
                [1, 0],
                format_func=yes_no,
            )
            estimated_salary = st.number_input(
                "Estimated salary",
                min_value=0.0,
                value=101348.88,
                step=100.0,
            )

        submitted = st.form_submit_button("Predict churn")

    payload = {
        "CreditScore": int(credit_score),
        "Geography": geography,
        "Gender": gender,
        "Age": int(age),
        "Tenure": int(tenure),
        "Balance": float(balance),
        "NumOfProducts": int(num_products),
        "HasCrCard": int(has_card),
        "IsActiveMember": int(active_member),
        "EstimatedSalary": float(estimated_salary),
    }

    return payload, submitted


def yes_no(value: int) -> str:
    """Format binary feature values for Streamlit select boxes."""
    return "Yes" if value == 1 else "No"


def render_model_summary(model_info: dict[str, Any] | None) -> None:
    """Render compact model metadata."""
    if model_info is None:
        st.warning(f"FastAPI service is not reachable at `{API_BASE_URL}`.")
        return

    metrics = model_info.get("metrics", {})
    first, second, third, fourth = st.columns(4)
    first.metric("Best model", model_info.get("best_model", "unknown"))
    second.metric("ROC-AUC", metrics.get("roc_auc", "n/a"))
    third.metric("Accuracy", metrics.get("accuracy", "n/a"))
    fourth.metric("F1-score", metrics.get("f1_score", "n/a"))


def render_prediction_result(result: dict[str, Any]) -> None:
    """Render the latest prediction result."""
    st.subheader(result["label"])

    left, right, third = st.columns(3)
    left.metric("Probability", result["probability"])
    right.metric("Prediction", result["prediction"])
    third.metric("Model version", result["model_version"])

    st.json(result)


def render_history() -> None:
    """Render recent prediction logs."""
    history = load_prediction_history(limit=25)
    if not history:
        st.info("Prediction history is not available yet.")
        return

    items = history.get("items", [])
    if not items:
        st.info("No predictions have been logged yet.")
        return

    frame = pd.DataFrame(items)
    st.dataframe(frame, use_container_width=True, hide_index=True)


def render_model_details(model_info: dict[str, Any] | None) -> None:
    """Render detailed model metadata."""
    if model_info is None:
        st.info("Model metadata is unavailable.")
        return

    st.json(model_info)


def main() -> None:
    """Run the Streamlit dashboard."""
    st.set_page_config(
        page_title="BankChurnPredict",
        page_icon=None,
        layout="wide",
    )

    st.title("BankChurnPredict")
    st.caption("Customer churn prediction dashboard")

    with st.sidebar:
        st.header("Service")
        st.text_input("API base URL", value=API_BASE_URL, disabled=True)

        if st.button("Refresh data"):
            st.cache_data.clear()
            st.rerun()

    model_info = load_model_info()
    render_model_summary(model_info)

    predict_tab, history_tab, model_tab = st.tabs(["Predict", "History", "Model"])

    with predict_tab:
        payload, submitted = build_prediction_payload()
        if submitted:
            try:
                result = post_json("/predict", payload)
                load_prediction_history.clear()
                render_prediction_result(result)
            except requests.RequestException as exc:
                st.error(f"Prediction request failed: {exc}")

    with history_tab:
        render_history()

    with model_tab:
        render_model_details(model_info)


if __name__ == "__main__":
    main()
