import collections
from abc import ABC, abstractmethod

import h2o
import numpy as np
import pandas as pd
from flaeming import __models__
from flaeming.data.static_variables import LAE_CLASS
from h2o.automl import H2OAutoML
from h2o.exceptions import H2OConnectionError
from sklearn.preprocessing import normalize


class AutoML(ABC):
    def __init__(
        self, dataframe: pd.DataFrame, feature_columns: list[str], target: str
    ):
        self.dataframe = dataframe
        self.ml_table = dataframe.loc[:, feature_columns + [target]]
        self.target = target

        try:
            h2o.connect()
        except H2OConnectionError:
            h2o.init()

    @abstractmethod
    def train(self, **kwargs):
        pass

    @abstractmethod
    def get_performance_data():
        pass

    def get_automl_params(self):
        def valid_entry(value):
            if isinstance(value, (int, float, str, list)):
                return True
            else:
                return False

        full_dict = vars(self.automl)["_H2OAutoML__input"]

        return {key: value for key, value in full_dict.items() if valid_entry(value)}

    def model_test_performance(self):
        return self.get_performance_data()

    def save_model(self, path: str):
        h2o.save_model(self.automl.leader, path)
        return

    @staticmethod
    def preprocess_features(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        table.loc[:, columns] = normalize(table.fillna(0).loc[:, columns])
        return table


class AutoMLRegressor(AutoML):
    def train(self, **kwargs):
        h20_df = h2o.H2OFrame(self.ml_table)

        self.train_df, self.test_df = h20_df.split_frame(ratios=[0.75], seed=3)

        self.automl = H2OAutoML(
            seed=3,
            # include_algos=["GBM", "DRF", "XGBoost", "GLM", "DeepLearning"],
            **kwargs,
        )

        self.automl.train(
            x=h20_df.columns[:-1], y=self.target, training_frame=self.train_df
        )
        return

    def get_performance_data(self):
        model_performance = self.automl.leader.model_performance(self.test_df)

        data = collections.defaultdict()

        data["r2"] = model_performance.r2()
        data["mae"] = model_performance.mae()
        data["mean_residual_deviance"] = model_performance.mean_residual_deviance()
        data["mse"] = model_performance.mse()
        data["rmse"] = model_performance.rmse()
        data["rmsle"] = model_performance.rmsle()

        for key, value in data.items():
            if isinstance(value, str):
                data[key] = -999

        return data


class AutoMLClassifier(AutoML):
    def train(self, **kwargs):
        h20_df = h2o.H2OFrame(self.ml_table)
        # Because it is a binary classification problem we need this
        h20_df[self.target] = h20_df[self.target].asfactor()

        self.train_df, self.test_df = h20_df.split_frame(ratios=[0.75], seed=3)

        self.automl = H2OAutoML(
            seed=3,
            # max_models = 3,
            # stopping_rounds=3,
            # exclude_algos = ["StackedEnsemble"],
            # include_algos=["GBM", "DRF", "XGBoost", "GLM", "DeepLearning"],
            # max_runtime_secs=3600,
            **kwargs,
        )

        self.automl.train(
            x=h20_df.columns[:-1], y=self.target, training_frame=self.train_df
        )
        return

    def get_performance_data(
        self,
    ):
        def get_metrics(prefix, model_performance):
            data = collections.defaultdict()
            data[f"{prefix}threshold"], data[f"{prefix}f1"] = model_performance.F1()[0]
            data[f"{prefix}accuracy"] = model_performance.accuracy(
                data[f"{prefix}threshold"]
            )[0][1]
            data[f"{prefix}precision"] = model_performance.precision(
                data[f"{prefix}threshold"]
            )[0][1]
            data[f"{prefix}recall"] = model_performance.recall(
                data[f"{prefix}threshold"]
            )[0][1]
            data[f"{prefix}auc"] = model_performance.auc()
            data[f"{prefix}aucpr"] = model_performance.aucpr()
            return data

        test_performance = self.automl.leader.model_performance(self.test_df)
        test_metrics = get_metrics("test_", test_performance)

        train_performance = self.automl.leader.model_performance(self.train_df)
        train_metrics = get_metrics("train_", train_performance)

        save_metrics = ["accuracy", "auc", "f1", "precision", "recall", "pr_auc"]

        cross_val = (
            self.automl.leader.cross_validation_metrics_summary()
            .as_data_frame()
            .set_index("")
            .loc[:, "mean"]
            .to_dict()
        )

        cross_val_data = {
            f"cv_{metric}": value
            for metric, value in cross_val.items()
            if metric in save_metrics
        }
        return {**train_metrics, **cross_val_data, **test_metrics}


def train_once(
    sc4k_table: pd.DataFrame, cosmos_table, feature_columns, target: str = LAE_CLASS
):
    ml_table = pd.concat(
        [cosmos_table.loc[:, feature_columns], sc4k_table.loc[:, feature_columns]],
        axis=0,
        ignore_index=True,
    )

    h20_df = h2o.H2OFrame(ml_table.loc[:, feature_columns])
    # Because it is a binary classification problem we need this
    h20_df[target] = h20_df[target].asfactor()

    train_df, test_df = h20_df.split_frame(ratios=[0.75])

    aml = H2OAutoML(
        seed=1,
        # max_models = 3,
        stopping_rounds=3,
        # exclude_algos = ["StackedEnsemble"],
        include_algos=["GBM", "DRF", "XGBoost", "GLM", "DeepLearning"],
        max_runtime_secs=3600,
    )

    aml.train(x=h20_df.columns[:-1], y=target, training_frame=train_df)
    h2o.save_model(aml.leader, f"{__models__}/bootstrap")

    return aml.leader.model_performance(test_df)
