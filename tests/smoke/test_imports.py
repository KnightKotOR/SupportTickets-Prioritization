"""Проверка, что модули импортируются без ошибок."""

import pytest


@pytest.mark.smoke
def test_optuna_search_importable():
    from src.optuna_search import Objective, OptunaSearchCV

    assert Objective is not None
    assert OptunaSearchCV is not None


@pytest.mark.smoke
def test_objective_signature():
    import inspect

    from src.optuna_search import Objective

    sig = inspect.signature(Objective.__init__)
    params = set(sig.parameters)
    assert {"X", "y", "X_val", "y_val", "model_name"}.issubset(params)
