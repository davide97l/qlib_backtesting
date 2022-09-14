# Backtesting pipeline

## Running backtesting

The backtesting pipeline is implemented in the file `backtest_pipeline.py`.
The pipeline takes as input a Qlib configuration which defines how to preprocess and segment the data, alphas, model hyperparameters, backtesting options, and more.
An example of configuration is defined [here](../configs/config_lightgbm_cohlv_csi500.py) as a Python `dict`.

You can quickly run the backtesting pipeline with:
```
python backtesting/backtest_pipeline.py --config configs/config_lightgbm_cohlv_csi500.py --save_dir backtest_results
```
`config_lightgbm_cohlv_csi500.py` is a simple config that leverages common trading features: CLOSE, OPEN, HIGH, LOW, OPEN, to train a LGBM model to predict the return of each stock of the `T+2` trading day.
Backtesting is performed with a [TopkDropoutStrategy](https://qlib.readthedocs.io/en/latest/component/strategy.html#topkdropoutstrategy) using the output of LGBM to rank stocks.
It is however possible to use different trading strategies chosen among the ones provided by [Qlib](https://qlib.readthedocs.io/en/latest/component/strategy.html) or defining your own.

Once execution terminates, the pipeline will generate a report containing the backtesting results and statistics, and save it in `save_dir`.
You can find an example of a report [here](../backtest_results/report_lightgbm_alphas158_csi500.html).

In general, you can replace `configs/your_config.py` with your own configuration.

## Data visualization and analysis

The data visualization and analysis pipeline is implemented in the file `data_visualization.py`.
This function could be useful to get some insights about your data such as variables relationships, correlations, missing values, etc.
It is particularly useful to analyze your own alphas.

If you simply need to visualize a portion of your dataset, you can execute the following command:
```
python backtesting/data_visualization.py --config configs/config_lightgbm_cohlv_csi500.py
```

In order to get a deeper visualization, you can run the same script specifying the name of a stock:
```
python backtesting/data_visualization.py --config configs/config_lightgbm_cohlv_csi500.py --stock SH600000
```
You can find an example of data visualization [here](../backtest_results/train_set_visualization_SH600000.html).

## Defining own alphas

If you need to define your personalized alphas, you can simply add the alphas' formula and name in your config:
```
"feature": [
    [
        "$close", "Ref($close, -2)", "Mean($close, 7)", "$high-$low", 
        "(EMA($close, 12) - EMA($close, 26))/$close - EMA((EMA($close, 12) - EMA($close, 26))/$close, 9)/$close"
    ],
    ["CLOSE", "CLOSE2", "MEAN7", "DELTA", "MACD"],
],
```
For more information you can refer [here](https://qlib.readthedocs.io/en/latest/advanced/alpha.html#example).

## Testing own alphas

In order to test the effectiveness of your alphas, we have implemented a [model](../model/alphas_strategy.py) whose output is simply the value of your alphas.
To run the pipeline with this model you can refer [here](configs/single_alpha_configs/config_singlealpha_ma_csi500.py).
```
python backtesting/backtest_pipeline.py --config configs/single_alpha_configs/config_singlealpha_ma_csi500.py
```

## Parallel pipeline

Sometimes it may happen that you have multiple configs to backtest, as for example in the case of backtesting alpha.
The parallel backtesting pipeline defined in `parallel_backtest_pipeline.py` allows users to run multiple configs with a single command and in a parallel way.
```
python backtesting/parallel_backtest_pipeline.py --config_path configs/single_alpha_configs --save_dir batch_backtest_results --njobs 2
```
The argument `config_path` is the path where your configs are stored, and you can define the number of parallel jobs with `njobs`.

## Benchmark

Qlib already provides to [baselines](https://qlib.readthedocs.io/en/latest/component/data.html#qlib-format-data):
- [Alphas158](https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py#L140): ([report](../backtest_results/report_config_lightgbm_alphas158_csi500.html))
- [Alphas360](https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py#L47)