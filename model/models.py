from keras import layers
from keras import optimizers
from keras.models import Model
from keras_contrib import losses
from keras_contrib import metrics
from keras_contrib.layers import CRF

from model.hyper_parameters import HyperParameters


def lstm_crf(args: HyperParameters):
    input_layer = layers.Input(shape=(args.MAX_SENTENCE,))

    model = layers.Embedding(args.WORD_COUNT, args.DENSE_EMBEDDING,
                             embeddings_initializer="uniform",
                             input_length=args.MAX_SENTENCE)(input_layer)

    model = layers.Bidirectional(
        layers.LSTM(args.LSTM_UNITS, recurrent_dropout=args.LSTM_DROPOUT,
                    return_sequences=True))(model)

    model = layers.TimeDistributed(
        layers.Dense(args.DENSE_UNITS, activation="relu"))(model)

    crf_layer = CRF(units=args.TAG_COUNT)
    output_layer = crf_layer(model)

    ner_model = Model(input_layer, output_layer)

    loss = losses.crf_loss
    acc_metric = metrics.crf_accuracy
    opt = optimizers.Adam(learning_rate=0.001)

    ner_model.compile(optimizer=opt, loss=loss, metrics=[acc_metric])

    ner_model.summary()

    return ner_model
