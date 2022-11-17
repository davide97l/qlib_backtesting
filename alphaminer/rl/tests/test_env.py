from alphaminer.rl.env import DataSource, TradingEnv, TradingPolicy, Portfolio, ActionRecorder
from os import path as osp
from typing import Tuple, List
from qlib.data.dataset import DataHandler
from qlib.data import D
import os
import pandas as pd
import numpy as np
import qlib
import tempfile
import shutil


def get_data_path() -> str:
    dirname = osp.dirname(osp.realpath(__file__))
    return osp.realpath(osp.join(dirname, "data"))


qlib.init(provider_uri=get_data_path(), region="cn")


class SimpleDataHandler(DataHandler):
    """
    Fit qlib data to RL env.
    """
    def __init__(self,
                 instruments,
                 start_time,
                 end_time,
                 init_data=True,
                 fetch_orig=True):
        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self._get_feature_config()
                }
            },
        }
        super().__init__(instruments, start_time, end_time, data_loader,
                         init_data, fetch_orig)

    def _get_feature_config(self) -> Tuple[List[str], List[str]]:
        fields = ["$close", "$open", "$factor"]
        names = ["CLOSE", "OPEN", "FACTOR"]
        return fields, names


def test_data_source():
    ds = DataSource(start_date="2010-01-01",
                    end_date="2020-01-01",
                    market="csi500",
                    data_handler=SimpleDataHandler(D.instruments("csi500"),
                                                   start_time="2010-01-01",
                                                   end_time="2020-01-01"))

    # Check query
    data = ds.query_trading_data("2012-01-04")
    assert data.shape[0] == 2

    data = ds.query_trading_data("2019-01-04")
    assert data.shape[0] == 3


def test_portfolio():
    ds = DataSource(start_date="2010-01-01",
                    end_date="2020-01-01",
                    market="csi500",
                    data_handler=SimpleDataHandler(D.instruments("csi500"),
                                                   start_time="2010-01-01",
                                                   end_time="2020-01-01"))

    cash = 0
    pf = Portfolio(cash=cash)
    assert len(pf.positions) == 0

    date = "2012-06-15"
    code = "SH600006"

    pf.positions = pd.Series({code: 1000}, dtype=np.float64)

    price = ds.query_trading_series(date, [code], fields="close")
    assert pf.nav(price) > 0

    # Test suspended stocks with null value
    pf = Portfolio(cash=0)
    codes = ["SH600006", "SH600021"]
    pf.positions = pd.Series([1000, 1000], index=codes, dtype=np.float64)
    price = ds.query_trading_series(date="2011-11-01",
                                    instruments=codes,
                                    fields="close")
    old_nav = pf.nav(price)
    price = ds.query_trading_series(
        date="2011-11-02",  # The day with null value
        instruments=codes,
        fields="close")
    new_nav = pf.nav(price)
    assert new_nav / old_nav > 0.9

    # Test delisted stock that still exists in portfolio
    pf = Portfolio(cash=0)
    codes = ["SH600006", "SH600021", "SH600607"]
    pf.positions = pd.Series([1000, 1000, 1000], index=codes, dtype=np.float64)
    price = ds.query_trading_series(date="2011-11-01",
                                    instruments=codes,
                                    fields="close")
    assert pf.nav(
        price) < 3300  # Price of delisted stock will not include in nav


def test_trading_policy():
    ds = DataSource(start_date="2010-01-01",
                    end_date="2020-01-01",
                    market="csi500",
                    data_handler=SimpleDataHandler(D.instruments("csi500"),
                                                   start_time="2010-01-01",
                                                   end_time="2020-01-01"))
    pf = Portfolio(cash=800000)

    action = pd.Series({
        "SH600006": 1.1,  # Buy
        "SH600008": 0,
    })
    tp = TradingPolicy(data_source=ds)
    date = "2012-06-15"
    pf, log_change = tp.take_step(date, action, portfolio=pf)
    assert log_change > np.log(0.9) and log_change < np.log(1.1)
    old_nav = pf.nav(
        ds.query_trading_series(date,
                                pf.positions.index.tolist(),
                                fields="close"))

    date = "2012-10-08"
    new_action = pd.Series({
        "SH600006": 0.6,  # Buy
        "SH600008": 0.5,  # Buy
    })
    pf, log_change = tp.take_step(date, new_action, portfolio=pf)
    assert log_change > np.log(0.9) and log_change < np.log(1.1)
    new_nav = pf.nav(
        ds.query_trading_series(date,
                                pf.positions.index.tolist(),
                                fields="close"))
    assert new_nav / old_nav < 0.81


def test_trading_env():
    ds = DataSource(start_date="2011-11-01",
                    end_date="2011-11-08",
                    market="csi500",
                    data_handler=SimpleDataHandler(D.instruments("csi500"),
                                                   start_time="2010-01-01",
                                                   end_time="2020-01-01"))
    tp = TradingPolicy(data_source=ds)
    env = TradingEnv(data_source=ds, trading_policy=tp, max_episode_steps=5)
    obs = env.reset()
    assert obs.shape[1] > 1
    done = False
    rewards = []
    for _ in range(5):
        action = pd.Series(np.random.rand(obs.shape[0]), index=obs.index)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
        assert isinstance(reward, float)
    assert np.unique(rewards).shape[0] == 5
    assert done

    # Test recorder
    tempdir = osp.join(tempfile.gettempdir(), "records")
    os.mkdir(tempdir)
    try:
        recorder = ActionRecorder(tempdir)
        env = TradingEnv(data_source=ds,
                         trading_policy=tp,
                         max_episode_steps=5,
                         recorder=recorder)
        obs = env.reset()
        for _ in range(5):
            action = pd.Series(np.random.rand(obs.shape[0]), index=obs.index)
            env.step(action)
        env.reset(dump_records=True)
        record_files = os.listdir(tempdir)
        assert len(record_files) == 1
        df = pd.read_csv(osp.join(tempdir, record_files[0]), index_col=0)
        assert df.shape == (5, 2)
        assert "2011-11" in str(df.index[0])
    finally:
        if osp.exists(tempdir):
            shutil.rmtree(tempdir)
