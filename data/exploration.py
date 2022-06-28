import re
import config
from dataloader import _read_csv
import matplotlib.pyplot as plt
from operator import itemgetter


def data_stats(data_df):
    print("Total number of observations in the dataset: {:,}".format(
        data_df["sample"].nunique()))
    print("Total words in the dataset: {:,}".format(data_df.shape[0]))

    data_df["tag"].value_counts().plot(kind="bar", figsize=(10, 5))
    # plt.show()

    data_df['tag'].apply(lambda x:
                         re.sub(r'[BI]-', '', x)).value_counts().plot(
        kind="bar", figsize=(10, 5))

    word_counts = data_df.groupby('sample')['tag'].agg(["count"])
    word_counts = word_counts.rename(columns={"count": "Word count"})
    word_counts.hist(bins=10, figsize=(8, 6))
    # plt.show()
    MAX_SENTENCE = word_counts.max()[0]
    print("Longest observation in the corpus contains {} words.".format(
        MAX_SENTENCE))

    all_words = list(set(data_df["entity"].values))
    all_tags = list(set(data_df["tag"].values))

    print("Number of unique words: {}".format(data_df["entity"].nunique()))
    print("Number of unique tags : {}".format(data_df["tag"].nunique()))
    return all_words, all_tags


def _generate_word2index_mapping(words):
    word2index = {word: idx + 2 for idx, word in enumerate(all_words)}
    word2index["--UNKNOWN_WORD--"] = 0
    word2index["--PADDING--"] = 1
    index2word = {idx: word for word, idx in word2index.items()}

    test_word = "cinnamon"

    test_word_idx = word2index[test_word]
    test_word_lookup = index2word[test_word_idx]

    print("The index of the word {} is {}.".format(test_word, test_word_idx))
    print("The word with index {} is {}.".format(
        test_word_idx, test_word_lookup))


def _to_tuples(x):
    iterator = zip(x["entity"].values.tolist(),
                   x["tag"].values.tolist())
    return [(word, tag) for word, tag in iterator]

def _generate_tag2index_mapping(data_df):


    sentences = data_df.groupby("sample").apply(_to_tuples).tolist()

    print(sentences[0])


if __name__ == '__main__':
    df = _read_csv(fr"{config.DATA_PATH}\\IOB tagging", i_col=0)
    all_words, all_tags = data_stats(df)