from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
import qlib
import pandas as pd
from qlib.contrib.report import analysis_model, analysis_position
import importlib.util
import argparse
import os


def pipeline(config_path, save_dir=''):
    qlib.init()
    # load config
    config_name = str(config_path.split('/')[-1].split('.')[0])
    spec = importlib.util.spec_from_file_location(config_name, config_path)
    config = importlib.util.module_from_spec(spec)  # creates a new module based on spec
    spec.loader.exec_module(config)  # executes the module in its own namespace when a module is imported or reloaded.

    # model and dataset initiaiton
    model = init_instance_by_config(config.task["model"])
    dataset = init_instance_by_config(config.task["dataset"])

    # train and test the model
    with R.start(experiment_name="workflow"):
        model.fit(dataset)
        R.save_objects(trained_model=model)

        rec = R.get_recorder()
        rid = rec.id  # save the record id

        # Inference and saving signal
        sr = SignalRecord(model, dataset, rec)
        sr.generate()

    # backtest and analysis
    with R.start(experiment_name='exp', recorder_id=rid, resume=True):
        # signal-based analysis
        rec = R.get_recorder()
        sar = SigAnaRecord(rec)  # get IC, ICIR, Rank IC, Rank ICIR
        sar.generate()

        # portfolio-based analysis: backtest
        # get mean, std, annualized_return, information_ratio, max_drawdown
        par = PortAnaRecord(rec, config.port_analysis_config, "day")
        par.generate()

    # load recorder
    recorder = R.get_recorder(recorder_id=rid, experiment_name='exp')
    # load previous results
    pred_df = recorder.load_object("pred.pkl")
    report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
    analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
    plots = []

    # https://qlib.readthedocs.io/en/latest/component/report.html#graphical-result
    result = analysis_position.report_graph(report_normal_df, show_notebook=False)
    plots += result

    # https://qlib.readthedocs.io/en/latest/component/report.html#graphical-result
    result = analysis_position.risk_analysis_graph(analysis_df, report_normal_df, show_notebook=False)
    plots += result

    label_df = dataset.prepare("test", col_set="label")
    label_df.columns = ['label']
    pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)

    result = analysis_position.score_ic_graph(pred_label, show_notebook=False)
    plots += result

    result = analysis_model.model_performance_graph(pred_label, show_notebook=False)
    plots += result

    # save report to html
    os.makedirs(save_dir, exist_ok=True)
    report_name = 'report_{}.html'.format(config_name)
    report_path = os.path.join(save_dir, report_name)
    with open(report_path, 'w') as f:
        for r in plots:
            f.write(r.to_html(full_html=False, include_plotlyjs='cdn'))
    print('Report saved to:', report_path)


# usage example:
# python backtesting/backtest_pipeline.py --config backtesting/config_lightgbm_cohlv_csi500.py --save_dir backtest_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    args = parser.parse_known_args()[0]
    qlib.init()  # must init before running all the other commands
    pipeline(args.config, args.save_dir)
