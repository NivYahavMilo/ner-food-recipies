import pandas as pd

import config


class Preprocess:
    data_df = None
    all_words = None
    all_tags = None
    word2index = None
    index2word = None
    index2tag = None
    tag2index = None
    sentences = None
    max_sentence = None


    @classmethod
    def set_class_attributes(cls):
        cls.data_df = pd.read_csv(
            fr"{config.DATA_PATH}\\IOB tagging.csv",
            index_col=0)

        cls.all_words = list(set(cls.data_df["entity"].values))
        cls.all_tags = list(set(cls.data_df["tag"].values))

        print("Number of unique words: {}".format(
            cls.data_df["entity"].nunique()))
        print("Number of unique tags : {}".format(
            cls.data_df["tag"].nunique()))

    @classmethod
    def _define_max_sentence(cls):
        word_counts = cls.data_df.groupby('sample')['tag'].agg(["count"])
        word_counts = word_counts.rename(columns={"count": "Word count"})
        cls.max_sentence = word_counts.max()[0]


    @classmethod
    def _generate_word2index_mapping(cls):
        word2index = {word: idx + 2 for idx, word in enumerate(cls.all_words)}
        setattr(cls, "word2index", word2index)

        word2index["--UNKNOWN_WORD--"] = 0
        word2index["--PADDING--"] = 1
        index2word = {idx: word for word, idx in word2index.items()}
        setattr(cls, "index2word", index2word)
        test_word = "cinnamon"

        test_word_idx = word2index[test_word]
        test_word_lookup = index2word[test_word_idx]

        print("The index of the word {} is {}.".format(test_word, test_word_idx))
        print("The word with index {} is {}.".format(
            test_word_idx, test_word_lookup))

    @classmethod
    def _tag_2_index(cls):
        cls.tag2index = {tag: idx + 1 for idx, tag in enumerate(cls.all_tags)}
        cls.tag2index["--PADDING--"] = 0

        cls.index2tag = {idx: word for word, idx in cls.tag2index.items()}

    @staticmethod
    def __to_tuples(x):
        iterator = zip(x["entity"].values.tolist(),
                       x["tag"].values.tolist())
        return [(word, tag) for word, tag in iterator]

    @classmethod
    def _generate_sentence_tuples(cls):
        sentences = cls.data_df.groupby("sample").apply(cls.__to_tuples).tolist()
        setattr(cls, "sentences", sentences)

    @classmethod
    def flow(cls):
        cls.set_class_attributes()
        cls._define_max_sentence()
        cls._generate_word2index_mapping()
        cls._generate_sentence_tuples()
        cls._tag_2_index()

        return cls
