import pandas as pd
import numpy as np
from alphaminer.gpmining.my_functions import RandomNFunctions as randF
from alphaminer.gpmining.my_functions import NInDataFunctions as nindF
import gplearn.functions as gpF
from qlib.data.dataset.loader import QlibDataLoader
import qlib
import pytest


@pytest.fixture(scope="function", autouse=True)
def init_tests():
    qlib.init()


def random_dataset(days: int = 252):
    # NOT USED
    date = pd.date_range(start='1/1/2010', periods=days)
    close = np.random.uniform(95, 105, size=days)
    open = np.random.uniform(95, 105, size=days)
    df = pd.DataFrame(
        {'open': open,
         'close': close,
         'date': date
         })
    df.set_index('date', inplace=True)
    return df


def qlib_dataset(config=None, instruments=None, start_time='20190101', end_time='20190630'):
    if instruments is None:
        instruments = ['sh000300']
    if config is None:
        config = ([["$close", "$open", "$high", "$low", "$volume"], ['CLOSE', 'OPEN', 'HIGH', 'LOW', 'VOLUME']])
    qdl = QlibDataLoader(
        config=config)
    df = qdl.load(instruments=instruments, start_time=start_time, end_time=end_time)
    return df


class TestFunction:

    @pytest.mark.unittest
    @pytest.mark.parametrize("d", [1, 5, 10])
    def test_randn_functions(self, d):
        df = qlib_dataset()
        configs = [
            [randF._delay(df['OPEN'].values, d), f'Ref($open, {d})'],  # Qlib can read values outside of the scope of the dataset if available
            [gpF.add2(df['OPEN'].values, df['HIGH'].values), 'Add($open, $high)'],
            [gpF.mul2(df['OPEN'].values, df['HIGH'].values), 'Mul($open, $high)'],
            [randF._ts_stddev(df['OPEN'].values, d), f'Std($open, {d})'],
            [randF._ts_sum(df['OPEN'].values, d), f'Sum($open, {d})'],
            [randF._ts_mean(df['OPEN'].values, d), f'Mean($open, {d})'],
            # [randF._ts_rank(df['OPEN'].values), f'Rank($open, {d})'],  # alpha_ql = alpha_gp * 4
            [randF._ts_argmax(df['OPEN'].values, d) + 1, f'IdxMax($open, {d})'],
            [randF._ts_sum(randF._ts_stddev(randF._ts_mean(gpF.add2(df['OPEN'].values, df['HIGH'].values), d), d), d), f'Sum(Std(Mean(Add($open, $high), {d}), {d}), {d})'],
            [randF._ts_sum(randF._delay(randF._ts_argmax(df['LOW'].values, d) + 1, d), d), f"Sum(Ref(IdxMax($low, {d}), {d}), {d})"],
        ]
        for config in configs:
            alpha_gp = config[0]
            alpha_ql = qlib_dataset(config=[[config[1]], ['ALPHA']])['ALPHA'].values
            alpha_ql = np.nan_to_num(alpha_ql)
            print(alpha_gp)
            print(alpha_ql)
            assert np.allclose(alpha_gp[d*4:], alpha_ql[d*4:])


    @pytest.mark.unittest
    def test_nindata_functions():
        df = qlib_dataset()



