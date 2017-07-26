from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D, ZeroPadding1D
from keras.layers.pooling import GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers.core import Flatten
from keras.layers.recurrent import LSTM
from keras.layers import merge, Input
from keras import backend as K
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers.normalization import BatchNormalization

import logging

embedding_dims = 23


def get_lstm_model(num_amino_acids, max_sequence_size, max_num_functions):
    logging.info('Building LSTM model using functional API ...')
    logging.debug("Embedding dims = " + str(embedding_dims))

    input = Input(shape=(max_sequence_size,))

    embedding = Embedding(num_amino_acids, embedding_dims, input_length=max_sequence_size, dropout=0.2)(input)
    # x = BatchNormalization()(embedding)
    x = LSTM(200, dropout_W=0.2, dropout_U=0.2, return_sequences=True)(embedding)
    x = LSTM(200, dropout_W=0.2, dropout_U=0.2)(x)

    x = Dense(max_num_functions)(x)
    x = BatchNormalization()(x) # can also try to do this after the activation (didn't work)
    output = Activation('sigmoid')(x)


    model = Model([input], output)
    model.summary()
    return model


def get_cnn_model(num_amino_acids, max_sequence_size, max_num_functions, target_function):
    logging.info('Building CNN model using functional API ...')
    logging.debug("Embedding dims = " + str(embedding_dims))

    # no need to mention the num_amino_acids when using functional API
    input = Input(shape=(max_sequence_size,))

    embedding = Embedding(num_amino_acids, embedding_dims, input_length=max_sequence_size)(input)

    x = Convolution1D(250, 15, activation='relu', subsample_length=1)(embedding)
    x = Dropout(0.3)(x)

    x = Convolution1D(100, 15, activation='relu', subsample_length=1)(x)
    x = Dropout(0.3)(x)

    # x = ZeroPadding1D((0,20))(x)
    #
    # z = Convolution1D(100, 5, subsample_length=1)(embedding)
    # z = BatchNormalization()(z)
    # z = Activation('sigmoid')(z)
    # z = Dropout(0.5)(z)
    #
    # z = Convolution1D(50, 5, subsample_length=1)(z)
    # z = BatchNormalization()(z)
    # z = Activation('sigmoid')(z)
    # z = Dropout(0.5)(z)



    # residual connection
    # x = merge([x, z], mode='concat')


    x = GlobalAveragePooling1D()(x)
    # x = Flatten()(x)

    if target_function != '':
        output_nodes = max_num_functions# 2
        output_activation = 'sigmoid' # 'softmax'
    else:
        output_nodes = max_num_functions
        output_activation = 'sigmoid'

    x = Dense(output_nodes)(x)
    x = BatchNormalization()(x) # can also try to do this after the activation (didn't work)
    output = Activation(output_activation)(x)


    model = Model([input], output)
    model.summary()
    return model
