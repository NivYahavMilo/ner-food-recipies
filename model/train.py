import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from model.hyper_parameters import HyperParameters
from model.models import LSTM
from model.preprocess import Preprocess
from model.utils import _set_device
import numpy as np


def _prepare_dataset(data: Preprocess, device: torch.device):
    sentences = data.sentences
    X = []
    y = []
    for sentence in sentences:
        sequence = torch.LongTensor([data.word2index[word] for word, _ in sentence])
        label_seq = torch.LongTensor([data.tag2index[tag] for _, tag in sentence])

        X.append(sequence)
        y.append(label_seq)

    X_len = torch.LongTensor([len(seq) for seq in X])
    X = pad_sequence(X, batch_first=True, padding_value=0)
    y = pad_sequence(y, batch_first=True, padding_value=-1)

    return (
        X.to(device),
        X_len.to(device),
        y.to(device))


def _set_training_params(data: Preprocess, args: HyperParameters):
    args.MAX_SENTENCE = data.max_sentence
    args.WORD_COUNT = len(data.index2word)
    args.TAG_COUNT = len(data.tag2index)


def _train():
    args = HyperParameters()
    raw_data = Preprocess.flow()
    _set_training_params(raw_data, args)
    device = _set_device()
    X, X_len, y = _prepare_dataset(raw_data, device)

    (X_train, X_test,
     X_len_train, X_len_test,
     y_train, y_test) = train_test_split(X, X_len, y,
                                         test_size=0.2,
                                         random_state=42)

    model = LSTM(
        k_input=args.WORD_COUNT,
        k_embeddings=args.EMBEDDING_DIM,
        k_layers=1,
        k_hidden=args.LSTM_UNITS,
        k_class=args.TAG_COUNT,
        bi_directional=False,
        return_states=False
    )

    model.to(device)
    print(model)

    loss_func = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters())

    max_length = torch.max(X_len_train)
    permutation = torch.randperm(X_train.size()[0])
    losses = np.zeros(args.EPOCHS)

    for epoch in range(args.EPOCHS):
        for i in range(0, X_train.size()[0], args.BATCH_SIZE):
            indices = permutation[i:i + args.BATCH_SIZE]

            batch_x, batch_y = X_train[indices], y_train[indices]
            batch_x_len = X_len_train[indices]

            y_pred = model(batch_x, batch_x_len, max_length)
            loss = loss_func(y_pred.view(-1, args.TAG_COUNT), batch_y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses[epoch] = loss
    print(losses)
    torch.save(model.state_dict(), 'ner_lstm.pt')

if __name__ == '__main__':
    _train()

