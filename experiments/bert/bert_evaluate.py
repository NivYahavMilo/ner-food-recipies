import os

from flair.data import Corpus
from flair.datasets import ColumnCorpus, DataLoader
from flair.models import SequenceTagger
from flair.visual.training_curves import Plotter

import config


def evaluate():

    columns = {0: 'text', 1: 'ner'}
    # init a corpus using column format data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus(config.DATA_PATH, columns,
                                  train_file='train.txt',
                                  test_file='test.txt',
                                  dev_file='dev.txt')

    eval_data = DataLoader(corpus.test, batch_size=1)


    ner_model = SequenceTagger.load(os.path.join(config.BERT_PATH,
                                                       "final-model.pt"))

    results, score = ner_model.evaluate(eval_data,
                                        out_path=fr"{config.BERT_PATH}\\predictions.txt")

    # visualize
    plotter = Plotter()
    plotter.plot_training_curves(os.path.join(config.BERT_PATH,
                                              'loss.tsv'))


if __name__ == '__main__':
    evaluate()