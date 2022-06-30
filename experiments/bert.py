import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification

import config


def preprocess_data():
    data = pd.read_csv(fr"{config.DATA_PATH}\\IOB tagging.csv", index_col=0)
    print(data.count())

    print("Number of tags: {}".format(len(data.tag.unique())))
    frequencies = data.tag.value_counts()
    print(frequencies)

    tags = {}
    for tag, count in zip(frequencies.index, frequencies):
        if tag[2:] not in tags.keys():
            tags[tag[2:]] = count
        else:
            tags[tag[2:]] += count

    print(sorted(tags.items(), key=lambda x: x[1], reverse=True))

    labels_to_ids = {k: v for v, k in enumerate(data.tag.unique())}
    ids_to_labels = {v: k for v, k in enumerate(data.tag.unique())}
    # fill missing values based on the last upper non-nan value
    data = data.fillna(method='ffill')

    # let's create a new column called "sentence" which groups the words by sentence
    data['sentence'] = data[['sample', 'entity', 'tag']].groupby(['sample'])[
        'entity'].transform(lambda x: ' '.join(x))
    # let's also create a new column called "word_labels" which groups the tags by sentence
    data['word_labels'] = data[['sample', 'entity', 'tag']].groupby(['sample'])[
        'tag'].transform(lambda x: ','.join(x))


    data = data[["sentence", "word_labels"]].drop_duplicates().reset_index(drop=True)
    print(data.head())




if __name__ == '__main__':

    preprocess_data()