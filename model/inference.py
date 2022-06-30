import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from model.hyper_parameters import HyperParameters
from model.preprocess import Preprocess
from model.train import _set_training_params, _set_device, _prepare_dataset
from model.utils import _get_mask
from model.models import LSTM



def _test(model, X, X_len, y_true, labels):
    model.eval()
    max_length = torch.max(X_len)
    mask = _get_mask(X_len, max_length)
    outputs = model(X, X_len, max_length)
    _, y_hat = torch.max(outputs.data, 2)
    y_hat = y_hat[mask == True]
    y = y_true[mask == True]
    correct = (y_hat == y).sum().item()
    a = correct / (mask == True).sum()

    print("Accuarcy: %0.3f " % a)
    print(classification_report(y, y_hat,
                                labels=labels.values(),
                                zero_division=1))
    cm = confusion_matrix(y, y_hat)
    cm_display = ConfusionMatrixDisplay(cm)
    cm_display.plot()


def run():
    args = HyperParameters()
    raw_data = Preprocess.flow()
    _set_training_params(raw_data, args)
    device = _set_device()


    model = LSTM(
        k_input=args.WORD_COUNT,
        k_embeddings=args.EMBEDDING_DIM,
        k_layers=1,
        k_hidden=args.LSTM_UNITS,
        k_class=args.TAG_COUNT,
        bi_directional=False,
        return_states=False
    )