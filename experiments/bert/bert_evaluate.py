import os
import config

from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.visual.training_curves import Plotter

from matplotlib import pyplot as plt



def evaluate():
    columns = {0: 'text', 1: 'ner'}
    # init a corpus using column format data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus(config.DATA_PATH,
                                  column_format=columns,
                                  train_file='train.txt',
                                  test_file='test.txt',
                                  dev_file='dev.txt')

    corpus.make_label_dictionary('ner')
    ner_model = SequenceTagger.load(os.path.join(config.BERT_PATH,
                                                 "final-model.pt"))

    result = ner_model.evaluate(corpus.test,
                                gold_label_type='ner',
                                out_path=fr"{config.BERT_PATH}\\predictions.txt"
                                )

    print(result.detailed_results)

    # visualize
    plotter = Plotter()
    plotter.plot_training_curves(os.path.join(config.BERT_PATH,
                                              'loss.tsv'))
    plt.show()
