import pytest
from optuna import create_study
from optuna.samplers import TPESampler

from src.optuna_search import Objective


@pytest.fixture
def small_objective(sample_train, sample_test):
    X = sample_train.drop(columns=["priority_cat"])
    y = sample_train["priority_cat"]
    X_val = sample_test.drop(columns=["priority_cat"])
    y_val = sample_test["priority_cat"]
    return Objective(X, y, X_val, y_val, "XGBClassifier")


@pytest.mark.model
def test_objective_returns_float(small_objective):
    study = create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(small_objective, n_trials=5)
    assert isinstance(study.best_value, float)
    assert study.best_value <= 1.0


@pytest.mark.model
def test_objective_raises_on_unknown_model(sample_train):
    X = sample_train.drop(columns=["priority_cat"])
    y = sample_train["priority_cat"]
    obj = Objective(X, y, X_val=X.head(1), y_val=y.head(1), model_name="UnknownModel")
    with pytest.raises(ValueError, match="Unknown model"):
        obj.__call__(
            type(
                "T",
                (),
                {
                    "suggest_float": lambda *a, **k: 0.1,
                    "suggest_int": lambda *a, **k: 1,
                    "suggest_categorical": lambda *a, **k: "l2",
                    "number": 0,
                },
            )()
        )
