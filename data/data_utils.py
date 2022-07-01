import os
import pandas as pd
import config
import warnings


pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)

ENTITIES = ["FOOD", "QUANTITY", "UNIT", "PROCESS", "PHYSICAL_QUALITY", "COLOR",
            "TASTE", "PURPOSE", "PART"]


def _read_csv(dataset, i_col=None):
    data = pd.read_csv(
        os.path.join(config.DATA_PATH, f'{dataset}.csv'),
        index_col=i_col)
    return data
