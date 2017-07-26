import numpy as np
np.random.seed(1337)  # for reproducibility, needs to be the first 2 lines in a script

import logging

from keras.models import Model, load_model
from train import prepare_data, predict_and_eval
from keras import metrics
from utils import get_protein_sequence

batch_size = 500

def predict_on_thresholds(model, X_train, y_train, X_test, y_test):
    thresholds = np.linspace(0.499, 0.503, 1)


    for threshold in thresholds:
        threhold = 0.5
        logs = predict_and_eval(model, X_train, y_train, X_test, y_test, threshold = threshold)

        metrics_line = 'threshold: %.5f - ' % threshold
        for s in ['loss', 'acc', 'precision', 'recall', 'fbeta_score']:
            metrics_line += "%s: %.5f %s: %.5f - " %(s, logs[s], 'val_'+s, logs['val_' +s])

        print metrics_line

if __name__ == '__main__':
    model_filename = "../results/00061-saved-model.h5"
    sequences_file = "../data/protein-seqs-2017-01-23-203946.txt"
    functions_file = "../data/protein-functions-2017-01-23-203946.txt"

    # reset logging config
    logging.basicConfig(format='%(asctime)s [%(levelname)7s] %(message)s', level=logging.DEBUG)
    X_train, y_train, X_test, y_test = prepare_data(sequences_file=sequences_file, functions_file=functions_file, target_function='0005524')

    print X_train.shape

    # load model
    model = load_model(model_filename)
    predict_on_thresholds(model, X_train, y_train, X_test, y_test)
