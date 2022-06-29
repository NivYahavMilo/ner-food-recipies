from keras import layers, Sequential
from keras import optimizers
from keras.models import Model

from model.hyper_parameters import HyperParameters


# def lstm_crf(args: HyperParameters):
#     input_layer = layers.Input(shape=(args.MAX_SENTENCE,))
#
#     model = layers.Embedding(args.WORD_COUNT, args.EMBEDDING_DIM,
#                              embeddings_initializer="uniform",
#                              input_length=args.MAX_SENTENCE)(input_layer)
#
#     model = layers.Bidirectional(
#         layers.LSTM(args.LSTM_UNITS, recurrent_dropout=args.LSTM_DROPOUT,
#                     return_sequences=True))(model)
#
#     model = layers.TimeDistributed(
#         layers.Dense(args.N_NEURONS, activation="relu"))(model)
#
#     crf_layer = CRF(units=args.TAG_COUNT)
#     output_layer = crf_layer(model)
#
#     ner_model = Model(input_layer, output_layer)
#
#     loss = losses.crf_loss
#     acc_metric = metrics.crf_accuracy
#     opt = optimizers.Adam(learning_rate=0.001)
#
#     ner_model.compile(optimizer=opt, loss=loss, metrics=[acc_metric])
#
#     ner_model.summary()
#
#     return ner_model






def LSTM(args):
    model = Sequential()
    model.add(layers.Embedding(
        input_dim=args.WORD_COUNT,
        output_dim=args.EMBEDDING_DIM,
        input_length=args.MAX_SENTENCE))

    model.add(layers.LSTM(
        units=args.EMBEDDING_DIM,
        input_dim=args.LSTM_UNITS,
        recurrent_dropout=args.LSTM_DROPOUT))

    model.add(layers.TimeDistributed(
        layer=layers.Dense(
        units=args.N_NEURONS,
        activation='relu')))


    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def BiLSTM(args: HyperParameters):
    input_word = layers.Input(shape=(args.MAX_SENTENCE,))
    model = layers.Embedding(
        input_dim=args.WORD_COUNT,
        output_dim=args.EMBEDDING_DIM,
        input_length=args.MAX_SENTENCE)(input_word)

    model = layers.SpatialDropout1D(
        rate=args.LSTM_DROPOUT)(model)

    model = layers.Bidirectional(layers.LSTM(
            units=args.LSTM_UNITS,
            recurrent_dropout=args.LSTM_DROPOUT))(model)

    out = layers.TimeDistributed(layers.Dense(
        units=args.N_NEURONS,
        activation="softmax"))(model)
    model = Model(input_word, out)
    model.summary()

