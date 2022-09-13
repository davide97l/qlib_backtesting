market = "csi300"  # or csi500
benchmark = "SH000300"  # must be one of the codes included in the market
train = ["2008-01-01", "2014-12-31"]
valid = ["2015-01-01", "2016-12-31"]
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
        "class": "CAPModel",
        "module_path": "qlib.contrib.model.capm",
        "kwargs": {
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
