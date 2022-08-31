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
from alphas.config_lightgbm_Alpha158 import data_handler_config, task, port_analysis_config


qlib.init()  # must init before running all the other commands

# model and dataset initiaiton
model = init_instance_by_config(task["model"])
dataset = init_instance_by_config(task["dataset"])

# train and test the model
with R.start(experiment_name="workflow"):
    model.fit(dataset)
    R.save_objects(trained_model=model)

    rec = R.get_recorder()
    rid = rec.id # save the record id

    # Inference and saving signal
    sr = SignalRecord(model, dataset, rec)
    sr.generate()

# backtest and analysis
with R.start(experiment_name='exp', recorder_id=rid, resume=True):
    # signal-based analysis
    rec = R.get_recorder()
    sar = SigAnaRecord(rec)  # get IC, ICIR, Rank IC, Rank ICIR
    sar.generate()

    #  portfolio-based analysis: backtest
    par = PortAnaRecord(rec, port_analysis_config,
                        "day")  # get mean, std, annualized_return, information_ratio, max_drawdown
    par.generate()

# load recorder
recorder = R.get_recorder(recorder_id=rid, experiment_name='exp')
# load previous results
pred_df = recorder.load_object("pred.pkl")
report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

# https://qlib.readthedocs.io/en/latest/component/report.html#graphical-result
analysis_position.report_graph(report_normal_df)

# https://qlib.readthedocs.io/en/latest/component/report.html#graphical-result
analysis_position.risk_analysis_graph(analysis_df, report_normal_df)

label_df = dataset.prepare("test", col_set="label")
label_df.columns = ['label']
pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)
# https://qlib.readthedocs.io/en/latest/component/report.html#graphical-result
analysis_position.score_ic_graph(pred_label)

# https://qlib.readthedocs.io/en/latest/component/report.html#graphical-result
analysis_model.model_performance_graph(pred_label)