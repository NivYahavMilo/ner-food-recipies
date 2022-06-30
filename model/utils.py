import torch
import numpy as np

K_SEED = 220

def _get_mask(X_len, max_length):
    mask = np.zeros((len(X_len), max_length))
    for ii, length in enumerate(X_len):
        mask[ii, :length] = 1

    return mask


def _set_device():
    global K_SEED
    torch.manual_seed(K_SEED)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    return device
