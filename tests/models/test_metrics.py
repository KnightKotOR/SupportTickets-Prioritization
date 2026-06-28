import pytest
from sklearn.metrics import confusion_matrix, f1_score


@pytest.mark.model
def test_f1_macro_above_threshold(trained_dummy_model, sample_test):
    X, y_true = sample_test.drop(columns=["priority_cat"]), sample_test["priority_cat"]
    y_pred = trained_dummy_model.predict(X)
    f1 = f1_score(y_true, y_pred, average="macro")
    assert f1 >= 0.85, f"F1-macro упал ниже порога: {f1:.4f}"


@pytest.mark.model
def test_no_high_to_low_misclassification(trained_dummy_model, sample_test):
    """Бизнес-требование: нет High → Low."""
    X, y_true = sample_test.drop(columns=["priority_cat"]), sample_test["priority_cat"]
    y_pred = trained_dummy_model.predict(X)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    # cm[i, j] - true=i, pred=j. High=3, Low=1 → cm[2,0] должно быть 0
    assert cm[2, 0] == 0, "Критическая ошибка: High-тикет классифицирован как Low"
