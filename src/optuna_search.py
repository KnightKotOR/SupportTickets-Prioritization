import warnings
from copy import deepcopy

import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


class Objective:
    """
    Целевая функция оптимизации гиперпараметров с кросс-валидацией
    """

    def __init__(self, X, y, X_val, y_val, model_name):
        self.X, self.y = X, y
        self.X_val, self.y_val = X_val, y_val

        self.model_name = model_name
        self.best_model = None
        self.best_val_score = float("-inf")

        self.n = []
        self.model_name_list = []
        self.F1_tuning_val_list = []
        self.F1_val_list = []
        self.params = []

    def __call__(self, trial):
        warnings.filterwarnings("ignore")

        clf_name = self.model_name
        if clf_name == "XGBClassifier":
            lr = trial.suggest_float("learning_rate", 1e-3, 5e-1, log=True)
            max_depth = trial.suggest_int("max_depth", 2, 12)
            min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
            gamma = trial.suggest_float("gamma", 0.0, 3.0)
            reg_alpha = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True)
            reg_lambda = trial.suggest_float("reg_lambda", 1e-4, 20.0, log=True)
            subsample = trial.suggest_float("subsample", 0.5, 1.0)
            colsample = trial.suggest_float("colsample_bytree", 0.5, 1.0)
            clf_obj = XGBClassifier(
                max_depth=max_depth,
                n_estimators=500,
                learning_rate=lr,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                subsample=subsample,
                colsample_bytree=colsample,
                min_child_weight=min_child_weight,
                objective="multi:softprob",
                gamma=gamma,
                eval_metric="mlogloss",
                n_jobs=-1,
            )
        elif clf_name == "LogisticRegression":
            C = trial.suggest_float("C", 1e-3, 10.0, log=True)
            max_iter = trial.suggest_int("max_iter", 100, 2000)
            penalty = trial.suggest_categorical("penalty", ["l1", "l2"])

            clf_obj = LogisticRegression(
                C=C,
                penalty=penalty,
                class_weight="balanced",
                max_iter=max_iter,
                n_jobs=-1,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model: {clf_name}")

        cv_ = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_tuning_cv_pred = cross_val_predict(clf_obj, self.X, self.y, cv=cv_, verbose=0, n_jobs=-1)
        clf_obj.fit(self.X, self.y)
        y_val_pred = clf_obj.predict(self.X_val)
        f1_tuning_cv_val = f1_score(self.y, y_tuning_cv_pred, average="macro")
        f1_val = f1_score(self.y_val, y_val_pred, average="macro")

        if f1_val > self.best_val_score:
            self.best_model = deepcopy(clf_obj)
            self.best_val_score = f1_val

        # Logging the results
        self.n.append(trial.number)
        self.model_name_list.append(clf_name)
        self.params.append(trial.params)
        self.F1_tuning_val_list.append(f1_tuning_cv_val)
        self.F1_val_list.append(f1_val)

        return f1_tuning_cv_val

    def get_results(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "n": self.n,
                "Model": self.model_name_list,
                "F1_tuning_val_list": self.F1_tuning_val_list,
                "F1_val": self.F1_val_list,
                "Parameters": self.params,
            }
        )


class OptunaSearchCV:
    """
    Класс для гиперпараметрического поиска по множеству моделей.
    В результате датафрейм с подробной информацией для анализа
    """

    def __init__(self, models_list):
        self.models_list = models_list
        self.best_models = []
        self.best_models_val = []

    def fit(self, x, y, x_val, y_val, n_trials=100, n_startup_trials=20):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        for model in self.models_list:
            print(f"\n{model} hyperoptimization")

            sampler = optuna.samplers.TPESampler(
                multivariate=True, n_startup_trials=n_startup_trials
            )

            objective = Objective(x, y, x_val, y_val, model)
            study = optuna.create_study(
                study_name=f"optimization_{model}", direction="maximize", sampler=sampler
            )
            study.set_metric_names(["F1_tuning_val"])
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            self.best_models.append(objective.best_model)
            self.best_models_val.append(objective.best_val_score)
            self.results_df = objective.get_results()
