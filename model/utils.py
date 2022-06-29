import numpy as np

def _get_mask(X_len, max_length):
    mask = np.zeros((len(X_len), max_length))
    for ii, length in enumerate(X_len):
        mask[ii, :length] = 1

    return mask

def _get_ner_accuracy():
    pass