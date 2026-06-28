import numpy as np
import pytest


@pytest.mark.invariant
def test_prediction_labels_in_valid_set(trained_dummy_model, sample_test):
    X = sample_test.drop(columns=["priority_cat"])
    preds = trained_dummy_model.predict(X)
    assert set(np.unique(preds)).issubset({0, 1, 2})


@pytest.mark.invariant
def test_critical_incident_never_low(trained_dummy_model, sample_test):
    """Тикет с is_critical_incident=1 не может быть Low."""
    critical = sample_test[sample_test["priority_cat"] == 2]
    if critical.empty:
        pytest.skip("Нет критичных инцидентов в выборке")
    preds = trained_dummy_model.predict(critical.drop(columns=["priority_cat"]))
    assert 0 not in preds, "Критичный инцидент предсказан как Low"


@pytest.mark.invariant
def test_model_deterministic(trained_dummy_model, sample_test):
    """Повторный предикт даёт тот же результат."""
    X = sample_test.drop(columns=["priority_cat"])
    p1 = trained_dummy_model.predict(X)
    p2 = trained_dummy_model.predict(X)
    np.testing.assert_array_equal(p1, p2)
