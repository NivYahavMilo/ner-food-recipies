import pickle
import config
import numpy as np
import torch
import torch.nn as nn

from experiments.rnn.hyper_parameters import HyperParameters
from model.lstm_sequence_tagger import LstmSequenceTagger
from experiments.rnn.preprocess import Preprocess
from experiments.rnn.utils import _set_device, _prepare_dataset, _get_mask, _plot_loss

from matplotlib import pyplot as plt

from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             ConfusionMatrixDisplay)

from sklearn.model_selection import train_test_split


def _set_training_params(data: Preprocess, args: HyperParameters):
    args.MAX_SENTENCE = data.max_sentence
    args.WORD_COUNT = len(data.index2word)
    args.TAG_COUNT = len(data.tag2index)


def _train(params):
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

    (X_train, X_val,
     X_len_train, X_len_val,
     y_train, y_val) = train_test_split(X_train, X_len_train, y_train,
                                        test_size=0.1,
                                        random_state=42)

    print("------INFO-------")
    print(f"Train size: {len(X_train)}")
    print(f"Validation size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")

    model = LstmSequenceTagger(
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

    max_length_train = torch.max(X_len_train)
    permutation_train = torch.randperm(X_train.size()[0])

    max_length_val = torch.max(X_len_val)
    permutation_val = torch.randperm(X_val.size()[0])

    losses = {'train_loss': np.zeros(args.EPOCHS),
              'val_loss': np.zeros(args.EPOCHS)}

    for epoch in range(args.EPOCHS):
        model.train()
        for i in range(0, X_train.size()[0], args.BATCH_SIZE):
            indices = permutation_train[i:i + args.BATCH_SIZE]

            batch_x, batch_y = X_train[indices], y_train[indices]
            batch_x_len = X_len_train[indices]

            y_pred = model(batch_x, batch_x_len, max_length_train)
            loss = loss_func(y_pred.view(-1, args.TAG_COUNT), batch_y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation mode
        model.eval()
        for i in range(0, X_val.size()[0], args.BATCH_SIZE):
            indices = permutation_val[i:i + args.BATCH_SIZE]

            batch_x, batch_y = X_val[indices], y_val[indices]
            batch_x_len = X_len_val[indices]

            y_pred = model(batch_x, batch_x_len, max_length_val)
            val_loss = loss_func(y_pred.view(-1, args.TAG_COUNT),
                                 batch_y.view(-1))

        losses['train_loss'][epoch] = loss
        losses['val_loss'][epoch] = val_loss

    print(losses['train_loss'])
    print(losses['val_loss'])

    # Saving loss of train and validation set
    with open(fr'{config.RNN_PATH}\\loss.pkl', 'wb') as handle:
        pickle.dump(losses, handle, protocol=pickle.HIGHEST_PROTOCOL)

    _plot_loss(losses)

    if eval(params["save"]):
        # Saving weights and biases
        torch.save(model.state_dict(), r'model\\ner_lstm.pt')
        print("model saved to pt")

    _evaluate(model=model,
              X_test=X_test,
              X_len_test=X_len_test,
              y_true=y_test,
              labels=raw_data.index2tag)


def _evaluate(model, X_test, X_len_test, y_true, labels):
    model.eval()
    max_length = torch.max(X_len_test)
    mask = _get_mask(X_len_test, max_length)
    outputs = model(X_test, X_len_test, max_length)
    _, y_hat = torch.max(outputs.data, 2)
    y_hat = y_hat[mask == True]
    y = y_true[mask == True]
    correct = (y_hat == y).sum().item()
    a = correct / (mask == True).sum()

    print("Accuarcy: %0.3f " % a)
    print(classification_report(y, y_hat))

    with open(fr'{config.RNN_PATH}\\classification report lstm.pkl', 'wb') as handle:
        pickle.dump(classification_report(
            y, y_hat, output_dict=True),
            handle, protocol=pickle.HIGHEST_PROTOCOL)

    cm = confusion_matrix(y, y_hat)
    cm_display = ConfusionMatrixDisplay(cm)
    cm_display.plot(cmap='plasma')
    plt.title("Confusion Matrix")
    plt.show()
