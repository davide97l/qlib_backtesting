# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
from typing import Text, Union
from qlib.data.dataset.weight import Reweighter

from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


class SingleAlpha(Model):
    """SingleAlpha
    This model outputs the value of a given alpha.
    Note that `fit` method is empty since no training is required.
    """

    def __init__(self, alpha: str, reverse_rank: bool = True):
        self.alpha = alpha
        self.reverse_rank = reverse_rank

    def fit(self, dataset: DatasetH, reweighter: Reweighter = None):
        return self

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        x_test_alpha = x_test[self.alpha]
        return pd.Series(x_test_alpha.values, index=x_test.index)
