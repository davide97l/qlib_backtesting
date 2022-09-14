from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
import qlib
import pandas as pd
from qlib.contrib.report import analysis_model, analysis_position
import importlib.util
import argparse
import os
from backtesting.backtest_pipeline import pipeline
from os import listdir
from os.path import isfile, join
from joblib import Parallel
import time

# usage example:
# python backtesting/parallel_backtest_pipeline.py --config_path backtesting/single_alpha_configs --save_dir batch_backtest_results --njobs 3


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--njobs', type=int, default=1)
    args = parser.parse_known_args()[0]
    # load configs
    config_path = args.config_path
    configs = [f for f in listdir(config_path) if isfile(join(config_path, f)) and f.endswith('.py')]
    start_time = time.time()
    result = Parallel(n_jobs=args.njobs)((pipeline,
             (os.path.join(config_path, config), args.save_dir,), {}) for config in configs)
    tot_time = time.time() - start_time
    print('Running time:', tot_time)
