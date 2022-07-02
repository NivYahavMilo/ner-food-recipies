import pickle

from colorama import Fore
import pandas as pd
from tabulate import tabulate

from model.utils import _plot_loss

labels = {'1': 'I-COLOR',
          '2': 'B-UNIT',
          '3': 'I-PHYSICAL_QUALITY',
          '4': 'I-PURPOSE',
          '5': 'I-PROCESS',
          '6': 'B-TASTE',
          '7': 'I-UNIT',
          '8': 'B-PHYSICAL_QUALITY',
          '9': 'I-TASTE',
          '10': 'B-QUANTITY',
          '11': 'B-FOOD',
          '12': 'B-PURPOSE',
          '13': 'I-FOOD',
          '14': 'B-COLOR',
          '15': 'I-QUANTITY',
          '16': 'I-PART',
          '17': 'B-PROCESS',
          '18': 'B-PART'}


def _load_results(name):
    with open(f'{name}.pkl', 'rb') as handle:
        return pickle.load(handle)


def _plot_classification_report(data):
    accuracy = round(data['accuracy'], 3)
    print(Fore.RED + f"total accuracy {accuracy}")
    data = pd.DataFrame.from_dict(data)
    data = data.drop(['accuracy', 'macro avg', 'weighted avg'], axis=1)
    data = data.rename(columns=labels)

    data = data.apply(lambda x: round(x, 2))
    print(Fore.GREEN + tabulate(data, headers='keys', tablefmt='pretty'))


def plot_results():
    loss = _load_results(name="loss")
    _plot_loss(loss)
    report = _load_results(name="classification report lstm")
    _plot_classification_report(report)


if __name__ == '__main__':
    plot_results()
