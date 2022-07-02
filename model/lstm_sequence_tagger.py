import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LstmSequenceTagger(nn.Module):

    def __init__(self, k_input, k_hidden, k_layers,
                 k_class, k_embeddings,
                 bi_directional=False,
                 return_states=False):
        super(LstmSequenceTagger, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=k_input,
            embedding_dim=k_embeddings,
            padding_idx=0)

        self.lstm = nn.LSTM(
            k_embeddings,
            k_hidden,
            k_layers,
            bidirectional=bi_directional,
            batch_first=True)

        self.fc = nn.Linear(k_hidden, k_class)

        self.return_states = return_states

    def forward(self, x, x_len, max_length=None):
        x = self.embedding(x)
        x = pack_padded_sequence(
            input=x,
            lengths=x_len,
            batch_first=True,
            enforce_sorted=False)

        x, _ = self.lstm(x)

        x, _ = pad_packed_sequence(
            x,
            batch_first=True,
            total_length=max_length)

        y = self.fc(x)

        if self.return_states:
            return x, y
        else:
            return y
