from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D, ZeroPadding1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.core import Flatten
from keras.layers import merge, Input
from keras import backend as K
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers.normalization import BatchNormalization

import logging

embedding_dims = 23

def get_cnn_model(num_amino_acids, max_sequence_size, max_num_functions):
    logging.info('Building CNN model using functional API ...')
    logging.debug("Embedding dims = " + str(embedding_dims))

    # no need to mention the num_amino_acids when using functional API
    input = Input(shape=(max_sequence_size,))

    embedding = Embedding(num_amino_acids, embedding_dims, input_length=max_sequence_size)(input)

    x = Convolution1D(200, 3, activation='relu', subsample_length=1)(embedding)
    x = Dropout(0.2)(x)

    z = Convolution1D(200, 3, activation='relu', subsample_length=1)(x)
    z = Dropout(0.2)(z)
    x = ZeroPadding1D((0,4))(x)

    # residual connection
    x = merge([z, x], mode='sum')


    x = GlobalMaxPooling1D()(x)

    x = Dense(max_num_functions)(x)
    x = BatchNormalization()(x) # can also try to do this after the activation (didn't work)
    output = Activation('sigmoid')(x)


    model = Model([input], output)
    model.summary()
    return model
