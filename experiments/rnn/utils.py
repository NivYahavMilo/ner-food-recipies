import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from torch.nn.utils.rnn import pad_sequence

K_SEED = 220

def _get_mask(X_len, max_length):
    mask = np.zeros((len(X_len), max_length))
    for ii, length in enumerate(X_len):
        mask[ii, :length] = 1

    return mask


def _prepare_dataset(data, device: torch.device):
    sentences = data.sentences
    X = []
    y = []
    for sentence in sentences:
        sequence = torch.LongTensor(
            [data.word2index[word] for word, _ in sentence])
        label_seq = torch.LongTensor(
            [data.tag2index[tag] for _, tag in sentence])

        X.append(sequence)
        y.append(label_seq)

    X_len = torch.LongTensor([len(seq) for seq in X])
    X = pad_sequence(X, batch_first=True, padding_value=0)
    y = pad_sequence(y, batch_first=True, padding_value=-1)

    return (
        X.to(device),
        X_len.to(device),
        y.to(device))

def _set_device():
    global K_SEED
    torch.manual_seed(K_SEED)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    return device


def _plot_loss(data):
    train_loss = data.get('train_loss')
    val_loss = data.get('val_loss')

    color_lst = ['black','cyan','blue', 'green',
                 'pink', 'red', 'violet', 'chocolate']
    correlations = []

    for _seq, _color in zip(data.items(), color_lst):
        mode, loss = _seq
        correlations.append({
            "name": mode,
            "x": loss.tolist(),
            "Y": [*range(50)],
            'color': colors.CSS4_COLORS[_color],
            'linewidth': 5,

        })

    fig, ax = plt.subplots(figsize=(20, 10), )

    for signal in correlations:
        ax.plot(signal['x'],  # signal['y'],
                color=signal['color'],
                linewidth=signal['linewidth'],
                label=signal['name'],
                )

    # Enable legend
    ax.legend(loc="upper right")
    ax.set_title("loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig(r"loss.png", dpi=100)



