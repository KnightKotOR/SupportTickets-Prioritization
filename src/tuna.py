from copy import deepcopy
import numpy as np
import optuna
import pandas as pd
import warnings

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.metrics import f1_score, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

warnings.filterwarnings('ignore')


class Objective(object):
    """
    Целевая функция оптимизации гиперпараметров с кросс-валидацией
    """

    def __init__(self, X, y, X_val, y_val, model_name):
        self.X, self.y = X, y
        self.X_val, self.y_val = X_val, y_val
        self.model_name = model_name
        self.model_results_df = pd.DataFrame(
            columns=['n', 'Model', 'F1_val', 'Parameters'])
        self.best_model = None
        self.best_model_est = None
        self.best_val_score = float('-inf')
        self.val_score = float('-inf')

    def __call__(self, trial):
        warnings.filterwarnings('ignore')
        # Определение модели и ее параметров
        # regressor_name = trial.suggest_categorical("regressor", self.models_list)
        model_name = self.model_name
        if model_name == "CatBoostClassifier":
            lr = trial.suggest_float("learning_rate", 0.01, 0.5, log=True)
            depth = trial.suggest_int("depth", 4, 10)
            l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True)
            bagging_temp = trial.suggest_float("bagging_temperature", 0.0, 2.0)
            random_strength = trial.suggest_float("random_strength", 0.1, 5.0)
            grow_policy = trial.suggest_categorical(
                "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]
            )
            #subsample = trial.suggest_float("subsample", 0.5, 1.0)

            classifier_obj = CatBoostClassifier(
                iterations=500,
                learning_rate=lr,
                depth=depth,
                l2_leaf_reg=l2_leaf_reg,
                bagging_temperature=bagging_temp,
                random_strength=random_strength,
                grow_policy=grow_policy,
                #subsample=subsample,
                #border_count=border_count,
                loss_function="MultiClass",
                auto_class_weights="Balanced",
                verbose=False,
                thread_count=4,
                early_stopping_rounds=5,
                random_state=42
            )
        elif model_name == "XGBClassifier":
            lr = trial.suggest_float("learning_rate", 0.01, 0.5, log=True)
            max_depth = trial.suggest_int("max_depth", 3, 12)
            min_child_weight = trial.suggest_float("min_child_weight", 1.0, 10.0, log=True)
            gamma = trial.suggest_float("gamma", 0.0, 3.0)
            reg_alpha = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True)
            reg_lambda = trial.suggest_float("reg_lambda", 1.0, 20.0, log=True)
            subsample = trial.suggest_float("subsample", 0.5, 1.0)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)

            classifier_obj = XGBClassifier(
                n_estimators=500,
                max_depth=max_depth,
                learning_rate=lr,
                min_child_weight=min_child_weight,
                gamma=gamma,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                objective="multi:softprob",
                eval_metric="mlogloss",
                tree_method="hist",
                n_jobs=-1,  
                random_state=42
            )
        elif model_name == "RandomForestClassifier":
            n = trial.suggest_int("n_estimators", 200, 1000)
            depth = trial.suggest_int("max_depth", 10, 40, log=True)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 30)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
            max_features = trial.suggest_float("max_features", 0.2, 0.7)
            bootstrap = trial.suggest_categorical("bootstrap", [True, False])
            oob_score = trial.suggest_categorical("oob_score", [True, False])

            classifier_obj = RandomForestClassifier(
                n_estimators=n,
                max_depth=depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                class_weight="balanced",
                bootstrap=bootstrap,
                oob_score=oob_score if bootstrap else False,
                n_jobs=-1,
                random_state=42
            )
        elif model_name == "HistGradientBoostingClassifier":
            learning_rate = trial.suggest_float("learning_rate", 0.02, 0.2, log=True)
            max_depth = trial.suggest_int("max_depth", 3, 12)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 10, 200)
            max_features = trial.suggest_float("max_features", 0.5, 1.0)
            l2_regularization = trial.suggest_float("l2_regularization", 1e-3, 1.0, log=True)
            max_bins = trial.suggest_categorical("max_bins", [2, 128, 255])

            classifier_obj = HistGradientBoostingClassifier(
                max_iter=500,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                l2_regularization=l2_regularization,
                early_stopping=True,
                n_iter_no_change=20,
                max_bins=max_bins,
                random_state=42
            )
        elif model_name == "LogisticRegression":
            C = trial.suggest_float("C", 1e-3, 10.0, log=True)
            max_iter = trial.suggest_int("max_iter", 100, 2000)
            penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
            if penalty == "l1": 
                solver = "saga"
            else:
                solver = trial.suggest_categorical("solver_l2", ["lbfgs", "saga", "liblinear"])

            classifier_obj = LogisticRegression(
                C=C,
                penalty=penalty,
                solver=solver,
                class_weight="balanced",
                max_iter=max_iter,
                n_jobs=-1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Кросс-валидация
        cv_ = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_tuning_pred = cross_val_predict(classifier_obj, self.X, self.y, cv=cv_, verbose=0, n_jobs=-1)
        y_val_pred = cross_val_predict(classifier_obj, self.X_val, self.y_val, cv=cv_, verbose=0, n_jobs=-1)
        f1_val = f1_score(self.y, y_tuning_pred, average="macro")
        f1_final_val = f1_score(self.y_val, y_val_pred, average="macro")

        if f1_final_val > self.best_val_score:
            self.best_model_est = deepcopy(classifier_obj)
            self.best_val_score = f1_final_val
            self.val_score = f1_val

        # Обновление дф
        self.model_results_df = pd.concat([self.model_results_df, pd.DataFrame({
            'n': trial.number,
            'Model': model_name,
            'F1_val': f1_final_val,
            'Parameters': [trial.params]
        })], ignore_index=True)

        return f1_val


class OptunaSearchCV:
    """
    Класс для гиперпараметрического поиска по множеству моделей.
    В результате датафрейм с подробной информацией для анализа
    """

    def __init__(self, models_list):
        self.models_list = models_list
        self.results_df = pd.DataFrame(columns=['n', 'Model', 'F1_val', 'Parameters'])
        self.best_models = []
        self.best_models_y_pred = {}
        self.best_models_val = []
        self.best_models_tuning_val = []
        self.opt_study_storage = optuna.storages.InMemoryStorage()

    def fit(self, x, y, x_val, y_val, n_trials=100, n_startup_trials=20):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        hyperopt_storage = optuna.storages.InMemoryStorage()
        for model in self.models_list:
            print(f"\n{model} hyperoptimization")

            objective = Objective(x, y, x_val, y_val, model)
            study = optuna.create_study(storage=hyperopt_storage, study_name=f"optimization_{model}",
                                        direction="maximize",
                                        sampler=optuna.samplers.TPESampler(multivariate=True,
                                                                           n_startup_trials=n_startup_trials))
            study.set_metric_names(["F1_val"])
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=-1)
            self.best_models.append(objective.best_model)
            self.best_models_val.append(objective.best_val_score)
            self.best_models_tuning_val.append(objective.val_score)
            self.results_df = pd.concat([self.results_df, objective.model_results_df], ignore_index=True)

