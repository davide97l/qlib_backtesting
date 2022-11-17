import gym
import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from os import path as osp
import os
from typing import List, Optional, Union, Dict, Tuple, Any
from qlib.data import D
from qlib.data.dataset import DataHandler
from time import time
from datetime import datetime


class DataSource:
    """
    A proxy between qlib and alphaminer rl framework.
    Please first download data from https://github.com/chenditc/investment_data,
    and init qlib by `qlib.init` before use this package.
    For fundamental data, use PIT data https://github.com/microsoft/qlib/tree/main/scripts/data_collector/pit/.
    """
    def __init__(self, start_date: Union[str, pd.Timestamp],
                 end_date: Union[str, pd.Timestamp], market: str,
                 data_handler: DataHandler) -> None:
        self._dates: List[pd.Timestamp] = D.calendar(
            start_time=start_date, end_time=end_date,
            freq='day').tolist()  # type: ignore
        self._market = market
        self._len_index = len(self.instruments(self._dates[0]))
        self._dh = data_handler
        self._obs_data = self._dh.fetch()
        self._trading_data = self._load_trading_data()
        self._benchmark_price = self._load_benchmark_price()
        self.label_in_obs = False

    def query_obs(self, date: Union[str, pd.Timestamp]) -> pd.DataFrame:
        """
        Get observations from data handler
        """
        if isinstance(date, str):
            date = pd.to_datetime(date)
        instruments = self.instruments(date)
        data = self._obs_data
        df = data[(data.index.get_level_values(1).isin(instruments))
                  & (data.index.get_level_values(0) == date)]
        df.reset_index(level=0, drop=True, inplace=True)
        #df = df.reindex(instruments)
        if not self.label_in_obs:
            labels = [col for col in df.columns if 'LABEL' in col.upper()]
            df = df.drop(labels, axis=1)
        # fill missing indexes with 0
        if len(df.index) < self._len_index:
            miss_indexes = set(instruments) - set(df.index)
            for miss_ind in miss_indexes:
                df.loc[miss_ind] = 0
                logging.info("Code {} {} is missing in obs!".format(
                    miss_ind, date))
        return df.fillna(0)

    def query_trading_data(
            self,
            date: Union[str, pd.Timestamp],
            instruments: Optional[List[str]] = None,
            fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Overview:
            Query trading data of the current day.
        Arguments:
            - date: the date.
            - instruments: the code in the query, if the parameter is None,
                will use the constituent stocks of the date.
            - fields: fields in list.
            - ffill: whether ffill the data when feature is none (useful in calculate nav).
        Example:
                           close      open    factor   close_1  suspended
            instrument
            SH600006    1.557522  1.587765  0.504052  1.582596      False
            SH600021    1.169540  1.220501  0.254802  1.205460      False
        """
        if isinstance(date, str):
            date = pd.to_datetime(date)
        if instruments is None:
            instruments = self.instruments(date)
        data = self._trading_data
        if len(instruments) == 0:
            df = pd.DataFrame(columns=data.columns)
        else:
            df = data[(data.index.get_level_values(0).isin(instruments))
                      & (data.index.get_level_values(1) == date)]
            df.reset_index(level=1, drop=True, inplace=True)
        return df if fields is None else df[fields]

    def query_trading_series(self,
                             date: Union[str, pd.Timestamp],
                             instruments: Optional[List[str]] = None,
                             fields: Optional[str] = None) -> pd.Series:
        """
        The helper function of query, which only query one field and return series instead of dataframe.
        """
        assert fields is not None, "Fields can not be null."
        return self.query_trading_data(date=date,
                                       instruments=instruments,
                                       fields=[fields])[fields]

    def _load_trading_data(self) -> pd.DataFrame:
        """
        Load data that is necessary for trading.
        """
        start = time()
        feature_map = {
            "$close": "close",
            "$open": "open",
            "$factor": "factor",
            "Ref($close,1)": "prev_close",
            "$close/$factor": "real_price",
        }
        codes = list(
            D.list_instruments(D.instruments(self._market),
                               start_time=self._dates[0],
                               end_time=self._dates[-1]).keys())
        # Need all the data to avoid suspended code in the market during the time window
        df = D.features(codes, list(feature_map.keys()), freq="day")
        df.rename(feature_map, axis=1, inplace=True)

        # Filter by chosen dates
        def processing_each_stock(df):
            code = df.index.get_level_values(0).unique()[0]
            df = df.loc[code]
            complete_dates = list(set(self._dates + list(df.index)))
            complete_dates.sort()
            # Append missing dates
            df = df.reindex(complete_dates)
            # If close is not in the dataframe, we think it is suspended or
            df["suspended"] = df["close"].isnull()
            df.fillna(method="ffill", inplace=True)
            # Trim into selected dates
            df = df[(df.index >= self._dates[0])
                    & (df.index <= self._dates[-1])]
            return df

        df = df.groupby(
            df.index.get_level_values(0)).apply(processing_each_stock)
        logging.warning(
            "Time cost: {:.4f}s | Init trading data Done".format(time() -
                                                                 start))
        return df

    def _load_benchmark_price(self) -> pd.DataFrame:
        benchmark_map = {"csi500": "SH000905"}
        benchmark = benchmark_map[self._market]
        feature_map = {"$close": "close", "Ref($close,1)": "prev_close"}
        df = D.features([benchmark],
                        list(feature_map.keys()),
                        freq="day",
                        start_time=self._dates[0],
                        end_time=self._dates[-1])
        df.rename(feature_map, axis=1, inplace=True)
        df["log_change"] = np.log(df["close"] / df["prev_close"])
        df.reset_index(level=0, drop=True, inplace=True)
        return df

    def query_benchmark(self, date: Union[str, pd.Timestamp]) -> pd.Series:
        if isinstance(date, str):
            date = pd.to_datetime(date)
        return self._benchmark_price.loc[date]

    def instruments(self, date: Union[str, pd.Timestamp]) -> List[str]:
        """
        Overview:
            Get instruments in the index.
        Arguments:
            - date: the date.
        """
        return D.list_instruments(D.instruments(self._market),
                                  start_time=date,
                                  end_time=date,
                                  as_list=True)

    @property
    def dates(self) -> List[pd.Timestamp]:
        return self._dates

    def next_date(self, date: Union[str, pd.Timestamp]) -> pd.Timestamp:
        if isinstance(date, str):
            date = pd.to_datetime(date)
        return self._dates[self._dates.index(date) + 1]

    def prev_date(self, date: Union[str, pd.Timestamp]) -> pd.Timestamp:
        if isinstance(date, str):
            date = pd.to_datetime(date)
        return self._dates[self._dates.index(date) - 1]


class Portfolio:
    """
    The portfolio contains positions and cash.
    """
    def __init__(self, cash: float = 1000000) -> None:
        self.cash = cash
        self.positions = pd.Series(dtype=np.float64)

    def nav(self, price: pd.Series) -> float:
        """
        Get nav of current portfolio.
        """
        assert not price.isnull().values.any(
        ), "Price contains null value when calculating nav, {}".format(price)

        miss_codes = set(self.positions.index) - set(price.index)
        if len(miss_codes) > 0:
            logging.warning(
                "Codes {} are missing in price when calculating nav.".format(
                    miss_codes))

        nav = (self.positions * price).sum() + self.cash
        return nav

    def __repr__(self) -> str:
        return "Cash: {:.2f}; Positions: {}".format(self.cash,
                                                    self.positions.to_dict())


class TradingPolicy:
    """
    A super naive policy which will buy the top 10 stocks
    in the action list, and sell the other stocks.
    This class will also focus on some trading rules like:
    - The cost of buy and sell.
    - The stop limit of buy and sell.
    - Round the number of shares to the board lot size.
    """
    def __init__(self,
                 data_source: DataSource,
                 buy_top_n: int = 50,
                 use_benchmark: bool = True) -> None:
        self._ds = data_source
        self._buy_top_n = buy_top_n
        self._stamp_duty = 0.001  # Charged only when sold.
        self._commission = 0.0003  # Charged at both side.
        self._use_benchmark = use_benchmark  # Use excess income to calculate reward.

    def take_step(self, date: Union[str, pd.Timestamp], action: pd.Series,
                  portfolio: Portfolio) -> Tuple[Portfolio, float]:
        """
        Overview:
            Take step, update portfolio and get reward (the change of nav).
            The default policy is buy the top 10 stocks from the action and sell the others
            at the open price of the t+1 day, then calculate the change of nav by the close
            price of the t+1 day.
        Arguments:
            - date: the date to take action.
            - action: the action.
        Returns:
            - portfolio, log_change: the newest portfolio and returns.
        """
        buy_stocks = action[action > 0].sort_values(
            ascending=False).index[:self._buy_top_n].tolist()
        prev_date = self._ds.prev_date(date)
        prev_price = self._ds.query_trading_series(
            prev_date, portfolio.positions.index.tolist(), fields="close")
        prev_nav = portfolio.nav(price=prev_price)  # type: ignore

        # Sell stocks
        for code, volume in portfolio.positions.items():
            if code in buy_stocks:
                continue
            value, hold = self.order_target_value(date, code, 0,
                                                  volume)  # type: ignore
            if hold == 0:
                portfolio.positions.drop(code, inplace=True)
            else:
                portfolio.positions.loc[code] = hold  # type: ignore
            portfolio.cash += value

        # Buy stocks
        if len(buy_stocks) > 0:
            open_price = self._ds.query_trading_series(
                date, portfolio.positions.index.tolist(), fields="open")
            current_nav = portfolio.nav(open_price)
            buy_value = current_nav / len(buy_stocks)
            for code in buy_stocks:
                volume = 0
                if code in portfolio.positions.index:
                    volume = portfolio.positions.loc[code]
                cash, hold = self.order_target_value(date, code, buy_value,
                                                     volume)  # type: ignore
                portfolio.cash += cash
                portfolio.positions.loc[code] = hold

        # Calculate reward
        future_price = self._ds.query_trading_series(
            date, portfolio.positions.index.tolist(), fields="close")
        nav = portfolio.nav(price=future_price)  # type: ignore
        log_change = np.log(nav / prev_nav)

        if self._use_benchmark:
            benchmark = self._ds.query_benchmark(date=date)
            benchmark_change = benchmark.loc["log_change"]
            log_change -= benchmark_change
        return portfolio, log_change

    def order_target_value(self, date: Union[str, pd.Timestamp], code: str,
                           value: float, hold: float) -> Tuple[float, float]:
        """
        Overview:
            Set an order into the market, will calculate the cost of trading.
        Arguments:
            - date: the date of order.
            - code: stock code.
            - value: value of cash.
            - hold: hold volume in current portfolio.
        Returns:
            - value, hold: change of cash and hold volume
        """
        # Sell or buy at the open price
        data = self._ds.query_trading_data(
            date, [code], ["open", "factor", "suspended"]).loc[code]
        open_price, factor, suspended = data.loc["open"], data.loc[
            "factor"], data.loc["suspended"]
        if suspended:
            return 0, hold
        # Trim volume by real open price, then adjust by factor
        volume = self._round_lot(code, value, open_price / factor) / factor
        # type: ignore
        cash = 0
        if hold > volume:  # Sell
            if self._available_to_sell(date, code):
                cash = open_price * (hold - volume) * (
                    1 - self._stamp_duty - self._commission)  # type: ignore
                hold = volume
            else:
                logging.warning("Stock {} {} is not available to sell.".format(
                    code, date))
        else:  # Buy
            if self._available_to_buy(date, code):
                cash = -open_price * (volume - hold) * (1 + self._commission
                                                        )  # type: ignore
                hold = volume
            else:
                logging.warning("Stock {} {} is not available to sell.".format(
                    code, date))
        return cash, hold

    def _available_to_buy(self, date: Union[str, pd.Timestamp],
                          code: str) -> bool:
        """
        Overview:
            Check if it is available to buy the stock.
            Possible reasons include suspension, non-trading days and others.
        """
        data = self._ds.query_trading_data(
            date, [code], fields=["open", "suspended", "prev_close"]).loc[code]
        open_price, suspended, prev_close = data.loc["open"], data.loc[
            "suspended"], data.loc["prev_close"]
        if suspended:
            return False
        if open_price / prev_close > (1 + self._stop_limit(code)):
            return False
        return True

    def _available_to_sell(self, date: Union[str, pd.Timestamp],
                           code: str) -> bool:
        data = self._ds.query_trading_data(
            date, [code], fields=["open", "suspended", "prev_close"]).loc[code]
        open_price, suspended, prev_close = data.loc["open"], data.loc[
            "suspended"], data.loc["prev_close"]
        if suspended:
            return False
        if open_price / prev_close < (1 - self._stop_limit(code)):
            return False
        return True

    def _round_lot(self, code: str, value: float, real_price: float) -> int:
        """
        Round the volume by broad lot.
        """
        if code[2:5] == "688":
            volume = int(value // real_price)
            if volume < 200:
                volume = 0
        else:
            volume = int(value // (real_price * 100) * 100)
        return volume

    def _stop_limit(self, code: str) -> float:
        if code[2:5] == "688" or code[2] == "3":
            return 0.195
        else:
            return 0.095


class Recorder(ABC):
    @abstractmethod
    def record(self, date: pd.Timestamp, value: pd.Series) -> None:
        raise NotImplementedError

    @abstractmethod
    def dump(self, file_name: Optional[str] = None):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError


class ActionRecorder(Recorder):
    def __init__(self, dirname: str = "./records") -> None:
        self._dirname = dirname
        self.reset()

    def record(self, date: pd.Timestamp, action: pd.Series) -> None:
        self._records["date"].append(date)
        self._records["action"].append(action)

    def dump(self, file_name: Optional[str] = None):
        if len(self._records["action"]) > 0 and len(self._records["date"]) > 0:
            df = pd.concat(self._records["action"],
                           axis=1,
                           keys=self._records["date"])
            df = df.transpose()
            if file_name is None:
                file_name = "action_{}.csv".format(
                    datetime.now().strftime("%y%m%d_%H%M%S"))
            path = osp.join(self._dirname, file_name)
            if not osp.exists(self._dirname):
                os.makedirs(self._dirname)
            df.to_csv(path)
            print('Record dumped at {}'.format(path))

    def reset(self):
        self._records = {"date": [], "action": []}


class TradingEnv(gym.Env):
    """
    Simulate all the information of the trading day.
    """
    def __init__(self,
                 data_source: DataSource,
                 trading_policy: TradingPolicy,
                 max_episode_steps: int = 20,
                 cash: float = 1000000,
                 recorder: Optional[Recorder] = None) -> None:
        super().__init__()
        self._ds = data_source
        self.max_episode_steps = max_episode_steps
        assert len(
            self._ds.dates
        ) > max_episode_steps, "Max episode step ({}) should be less than effective trading days ({}).".format(
            max_episode_steps, len(self._ds.dates))
        self._trading_policy = trading_policy
        self._cash = cash
        self.observation_space = np.array(
            self._ds.query_obs(
                date=self._ds.dates[0]).values.shape)  # type: ignore
        self.action_space = self.observation_space[0]  # number of instruments
        self.reward_range = (-np.inf, np.inf)
        self._recorder = recorder

        self._reset()

    def step(
        self, action: pd.Series
    ) -> Tuple[pd.DataFrame, float, bool, Dict[Any, Any]]:
        next_date = self._ds.next_date(self._today)
        self._portfolio, reward = self._trading_policy.take_step(
            next_date, action=action, portfolio=self._portfolio)
        obs = self._ds.query_obs(date=next_date)
        self._step += 1
        done = True if self._step >= self.max_episode_steps else False
        self._today = next_date
        if self._recorder:
            self._recorder.record(self._today, action)
        return obs, reward, done, {}

    def reset(self, dump_records: bool = False) -> pd.DataFrame:
        """
        Reset states and return the reset obs.
        """
        self._reset()
        if dump_records and self._recorder:
            self._recorder.dump()
        if self._recorder:
            self._recorder.reset()
        obs = self._ds.query_obs(self._today)
        return obs

    def _reset(self) -> None:
        """
        Reset states.
        """
        self._today = np.random.choice(
            self._ds.dates[:-self.max_episode_steps])  # type: ignore
        self._step = 0
        self._portfolio = Portfolio(cash=self._cash)

    def close(self) -> None:
        pass
