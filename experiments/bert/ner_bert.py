from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter

import config


def _train():
    columns = {0: 'text', 1: 'ner'}
    # init a corpus using column format data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus(config.DATA_PATH, columns,
                                  train_file='train.txt',
                                  test_file='test.txt',
                                  dev_file='dev.txt')

    # make the label dictionary from the corpus
    label_dict = corpus.make_label_dictionary(label_type='ner')
    print(label_dict)

    # initialize fine-tuneable transformer embeddings WITH document context
    embeddings = TransformerWordEmbeddings(model='bert-base-uncased',
                                           layers="-1",
                                           subtoken_pooling="first",
                                           fine_tune=True,
                                           use_context=True,
                                           )

    # initialize bare-bones sequence tagger
    tagger = SequenceTagger(hidden_size=64,
                            embeddings=embeddings,
                            tag_dictionary=label_dict,
                            tag_type='ner',
                            use_crf=False,
                            use_rnn=False,
                            reproject_embeddings=False)

    # initialize trainer
    trainer = ModelTrainer(tagger, corpus)

    # run fine-tuning
    trainer.fine_tune('experiments/bert',
                      learning_rate=5.0e-6,
                      mini_batch_size=4,
                      mini_batch_chunk_size=1)
    # visualize
    # plotter = Plotter()
    # plotter.plot_training_curves('loss.tsv')


