"""
Helper function for data preprocessing
"""

import os
import warnings

import pandas as pd

import config

pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)

ENTITIES = ["FOOD", "QUANTITY", "UNIT", "PROCESS", "PHYSICAL_QUALITY", "COLOR",
            "TASTE", "PURPOSE", "PART"]


def _read_csv(dataset, i_col=None):
    data = pd.read_csv(
        os.path.join(config.DATA_PATH, f'{dataset}.csv'),
        index_col=i_col)
    return data


def _split_data_to_train_test(test_size: int = 200):
    data = _read_csv("IOB tagging", i_col=0)
    n_samples = len(data['sample'].unique())
    train_size = n_samples - test_size


    train_data = data[data['sample'] <= train_size]
    test_data = data[data['sample'] > train_size]

    train_data.to_csv("IOB tagging train.csv")
    test_data.to_csv("IOB tagging test.csv")







