"""Валидация схемы датасета тикетов."""

import pandera.pandas as pa
import pytest
from pandera import Check, Column

SCHEMA = pa.DataFrameSchema(
    {
        "day_of_week_num": Column(int, checks=Check.isin([1, 2, 3, 4, 5, 6, 7])),
        "company_id": Column(int, checks=Check.ge(10000)),
        "industry_cat": Column(int, checks=Check.isin([0, 1, 2])),
        "customer_tier_cat": Column(int, checks=Check.isin([1, 2, 3, 4, 5, 6, 7])),
        "region_cat": Column(int, checks=Check.isin([1, 2, 3])),
        "past_30d_tickets": Column(int, checks=Check.ge(0)),
        "past_90d_incidents": Column(int, checks=Check.ge(0)),
        "product_area_cat": Column(int, checks=Check.isin([1, 2, 3, 4, 5, 6])),
        "booking_channel_cat": Column(int, checks=Check.isin([1, 2, 3, 4])),
        "reported_by_role_cat": Column(int, checks=Check.isin([1, 2, 3, 4, 5])),
        "customers_affected": Column(int, checks=Check.ge(0)),
        "error_rate_pct": Column(int, checks=Check.ge(0)),
        "downtime_min": Column(float, checks=Check.ge(0)),
        "payment_impact_flag": Column(int, checks=Check.isin(0, 1)),
        "security_incident_flag": Column(int, checks=Check.isin(0, 1)),
        "data_loss_flag": Column(int, checks=Check.isin(0, 1)),
        "has_runbook": Column(int, checks=Check.isin(0, 1)),
        "customer_sentiment_cat": Column(int, checks=Check.isin(0, 1, 2, 3)),
        "description_length": Column(int, checks=Check.in_range(20, 748)),
        "priority_cat": Column(int, checks=Check.isin([0, 1, 2])),
    },
    strict=True,
    coerce=True,
)


@pytest.mark.data
def test_schema(sample_dataset):
    SCHEMA.validate(sample_dataset)


@pytest.mark.data
def test_class_balance_within_bounds(sample_dataset):
    """README: Low ~50%, Medium ~35%, High ~15%."""
    counts = sample_dataset["priority"].value_counts(normalize=True)
    assert 0.40 < counts["Low"] < 0.60
    assert 0.25 < counts["Medium"] < 0.45
    assert 0.05 < counts["High"] < 0.25
