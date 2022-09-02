import qlib
from qlib.data.dataset.loader import QlibDataLoader
from qlib.data.dataset.processor import ZScoreNorm, Fillna, CSZScoreNorm, DropnaLabel
from qlib.data.dataset import DataHandlerLP

market = "csi300"  # or csi500
benchmark = "SH000300"  # must be one of the codes included in the market
train = ["2013-01-01", "2015-01-29"]
valid = ["2015-01-31", "2016-12-31"]
test = ["2017-01-01", "2020-08-01"]

data_handler_config = {
    "start_time": train[0],
    "end_time": test[1],
    "instruments": market,
    "data_loader": {
        "class": QlibDataLoader,
        "kwargs": {
            "config": {
                # all alphas operators: https://github.com/microsoft/qlib/blob/main/qlib/data/ops.py
                "feature": [["$close", "$open", "$high", "$low", "$volume"], ['CLOSE', 'OPEN', 'HIGH', 'LOW', 'VOLUME']],
                "label": [["Ref($close, -2)/Ref($close, -1) - 1"], ['LABEL0']],
                },
            "freq": "day",
            },
        },
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
                'class': DataHandlerLP,
                "module_path": qlib.data.dataset.handler,
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
