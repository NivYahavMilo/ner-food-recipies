import os

ROOT_PATH = os.path.abspath(os.path.curdir)
DATA_PATH = os.path.join(ROOT_PATH, "data")
EXPERIMENTS_PATH = os.path.join(ROOT_PATH, "experiments")
RNN_PATH = os.path.join(EXPERIMENTS_PATH, "rnn")
BERT_PATH = os.path.join(EXPERIMENTS_PATH, "bert")
FIGURES_PATH = os.path.join(ROOT_PATH, "figures")