"""Общие фикстуры для тестов SupportTickets-Prioritization."""

from pathlib import Path

import pandas as pd
import pytest

FIXTURES_DIR = Path("./data/fixtures/")

EXPECTED_FEATURES = [
    "day_of_week_num",
    "company_id",
    "company_size",
    "industry_cat",
    "customer_tier_cat",
    "region_cat",
    "past_30d_tickets",
    "past_90d_incidents",
    "product_area_cat",
    "booking_channel_cat",
    "reported_by_role_cat",
    "customers_affected",
    "error_rate_pct",
    "downtime_min",
    "payment_impact_flag",
    "security_incident_flag",
    "data_loss_flag",
    "has_runbook",
    "customer_sentiment_cat",
    "description_length",
    "priority_cat",
]
TARGET = "priority_cat"


@pytest.fixture(scope="session")
def sample_train() -> pd.DataFrame:
    """Маленький набор для быстрых тестов."""
    return pd.read_csv(FIXTURES_DIR / "dataset_sample.csv")


@pytest.fixture(scope="session")
def trained_dummy_model(sample_train):
    """Маленькая XGBoost-модель для инвариантных тестов."""
    from xgboost import XGBClassifier

    X = sample_train[EXPECTED_FEATURES]
    y = sample_train[TARGET]
    model = XGBClassifier(
        n_estimators=20,
        max_depth=3,
        learning_rate=0.1,
        n_jobs=1,
        random_state=42,
        eval_metric="mlogloss",
    )
    model.fit(X, y)
    return model


@pytest.fixture
def high_priority_ticket() -> pd.DataFrame:
    """Тикет высокого приоритета"""
    return pd.DataFrame(
        [
            {
                "day_of_week_num": 2,
                "company_id": 100018,
                "industry_cat": 3,
                "customer_tier_cat": 2,
                "region_cat": 2,
                "past_30d_tickets": 5,
                "past_90d_incidents": 2,
                "product_area_cat": 4,
                "booking_channel_cat": 4,
                "reported_by_role_cat": 4,
                "customers_affected": 51,
                "error_rate_pct": 5.988785895,
                "downtime_min": 44,
                "payment_impact_flag": 0,
                "security_incident_flag": 0,
                "data_loss_flag": 0,
                "has_runbook": 1,
                "customer_sentiment_cat": 2,
                "description_length": 206,
                "priority_cat": 2,
            }
        ]
    )
