import numpy as np
from plot_keras_history import plot_history
from sklearn.model_selection import train_test_split

from model.hyper_parameters import HyperParameters
from model.models import lstm_crf
from model.preprocess import Preprocess


def prepare_dataset(data: Preprocess, args):
    sentences = data.sentences
    X = [[word[0] for word in sentence] for sentence in sentences]
    y = [[word[1] for word in sentence] for sentence in sentences]

    args.MAX_SENTENCE = data.max_sentence
    args.WORD_COUNT = len(data.index2word)
    args.TAG_COUNT = len(data.tag2index)

    X = [[data.word2index[word] for word in sentence] for sentence in X]
    y = [[data.tag2index[tag] for tag in sentence] for sentence in y]

    X = [sentence + [data.word2index["--PADDING--"]] * (
                args.MAX_SENTENCE - len(sentence))
         for sentence in X]
    y = [sentence + [data.tag2index["--PADDING--"]] * (
                args.MAX_SENTENCE - len(sentence))
         for sentence in y]
    print("X[0]:", X[0])
    print("y[0]:", y[0])

    y = [np.eye(args.TAG_COUNT)[sentence] for sentence in y]
    print("X[0]:", X[0])
    print("y[0]:", y[0])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    print(
        "Number of sentences in the training dataset: {}".format(len(X_train)))
    print("Number of sentences in the test dataset : {}".format(len(X_test)))

    X_train = np.asarray(X_train).astype(float)
    X_test = np.asarray(X_test).astype(float)
    y_train = np.asarray(y_train).astype(float)
    y_test = np.asarray(y_test).astype(float)

    return X_train, X_test, y_train, y_test


def _train():
    args = HyperParameters()
    raw_data = Preprocess.flow()
    X_train, X_test, y_train, y_test = prepare_dataset(raw_data, args)
    ner_model = lstm_crf(args)
    history = ner_model.fit(X_train, y_train,
                        batch_size=args.BATCH_SIZE,
                        epochs=args.MAX_EPOCHS,
                        validation_split=0.1,
                        verbose=2)

    plot_history(history.history)


if __name__ == '__main__':
    _train()
