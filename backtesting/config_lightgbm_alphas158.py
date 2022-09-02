from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
import qlib
import pandas as pd
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import (
    backtest_daily as normal_backtest,
    risk_analysis,
)
from qlib.contrib.report import analysis_model, analysis_position
from qlib.data.dataset.loader import QlibDataLoader
from qlib.data.dataset.processor import ZScoreNorm, Fillna, CSZScoreNorm, DropnaLabel

market = "csi300"  # or csi500
benchmark = "SH000300"  # must be one of the codes included in the market
train = ["2013-01-01", "2015-01-29"]
valid = ["2015-01-31", "2016-12-31"]
test = ["2017-01-01", "2020-08-01"]

data_handler_config = {
    "start_time": train[0],
    "end_time": test[1],
    "fit_start_time": train[0],
    "fit_end_time": train[1],
    "instruments": market,
}
task = {
    "model": {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,
        },
    },
    "dataset": {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": data_handler_config,
            },
            "segments": {
                "train": tuple(train),
                "valid": tuple(valid),
                "test": tuple(test),
            },
        },
    },
}
port_analysis_config = {
    "executor": {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    },
    "strategy": {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "kwargs": {
            "signal": "<PRED>",
            "topk": 50,
            "n_drop": 5,
        },
    },
    "backtest": {
        "start_time": test[0],
        "end_time": test[1],
        "account": 100000000,
        "benchmark": benchmark,
        "exchange_kwargs": {
            "freq": "day",
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    },
}
