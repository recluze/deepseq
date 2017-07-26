import numpy as np
np.random.seed(1337)  # for reproducibility, needs to be the first 2 lines in a script

import json
from utils import *
import time
import os.path
from visualization import visualize_history, LossHistory


from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.datasets import imdb
from keras import backend as K
from keras.callbacks import CSVLogger, ModelCheckpoint

from keras.optimizers import SGD
from keras import metrics

from models import get_cnn_model, get_lstm_model

# set parameters:
num_amino_acids = 23 # alphabet of proteins +1 for padding
batch_size = 32 # 600 # tried: 10 (299s), 100 (542s), 32
nb_epoch = 50

max_num_functions = 5 # 855 # with cutoff 10. With everything: 4072

# target_function = '0005524' # ATP binding
# target_function = '0046872' # metal ion binding
# target_function = '0008270' # zinc ion binding
# target_function = '0003677' # DNA binding
# target_function = '0003676' # nucleic acid binding
# target_function = '0003700' # transcription factor activity, sequence-specific DNA binding;
# target_function = '0005509' # calcium ion binding

target_function = ''

max_sequence_size = 2000 # We're skipping anything longer than this. Absolute max is 34k.
samples_to_predict = 0 # 4655

# TODO: remove size restriction later.
restrict_sample_size = None
# skip_epochs_upto = -1 # deprecated

model_snapshot_directory = "../results"

def perform_data_split(X, y):
    # currently doing 2/3 for training and 1/3 for testing
    n = X.shape[0]

    # randomize to shuffle first
    randomize = np.arange(n)
    np.random.shuffle(randomize)

    X = X[randomize]
    y = y[randomize]

    # now split
    test_split = n * 2 / 3
    X_train = X[:test_split]
    y_train = y[:test_split]
    X_test  = X[test_split:]
    y_test  = y[test_split:]
    return (X_train, y_train, X_test, y_test)

def prepare_data(sequences_file, functions_file, target_function):
    logging.info("Preparing data ... ")

    # input protein-function map
    protein_function_map = {}
    with open(functions_file) as fn_file:
        protein_function_map = json.load(fn_file)

    # input all X
    sequences = []
    p = []
    X = []
    y = []

    pos_examples = 0
    neg_examples = 0

    with open(sequences_file) as f:
        for line in f:
            ln = line.split(',')
            protein_id = ln[0].strip()
            seq = ln[1].strip()

            # we're doing this to reduce input size
            if len(seq) >= max_sequence_size:
                continue

            try:
                # need to convert function to ASCII first
                functions = [fn.encode('ascii', 'ignore') for fn in protein_function_map[protein_id]]
                p.append(protein_id)
                X.append(seq)

                if target_function != '':
                    if target_function in functions:
                        y.append(1)
                        pos_examples += 1
                    else:
                        y.append(0)
                        neg_examples += 1
                else:
                    y.append(functions)


            except KeyError:
                pass # For some proteins, we don't have annotations. skip these

    logging.info("Got " + str(len(X)) + " data points. Example below: ")
    logging.info(p[0])
    logging.info(X[0])
    logging.info(y[0])
    logging.info("Pos examples: " + str(pos_examples) + " Neg examples: " + str(neg_examples))

    logging.info('')
    logging.info("Converting to vector representation...")

    logging.info("Maximum length of a sequence:" + str(max_sequence_size))
    X_all = []
    y_all = []

    br = 0
    for i in range(len(X)): # need to do this in loop to ensure ordering
        x = sequence_to_indices(X[i])

        # if needed to do this, use nputils.to_categorical
        # if target_function != '':
        #     y_i = np.array([1, 0]) if y[i] == 0 else np.array([0, 1])
        # else:
        #     y_i = functions_to_indices(y[i])

        y_i = y[i]

        # print y_i.nonzero()
        # print x
        X_all.append(x)
        y_all.append(y_i)


        br += 1
        if restrict_sample_size != None and br > restrict_sample_size: break

    X_all = np.array(X_all)
    y_all = np.array(y_all)
    # print X_all.shape
    # print y_all.shape

    # for protein_index in range(len(p)):
    #     print p[protein_index], '-', y_all[protein_index]

    logging.info('Padding sequences to %d ... ' % max_sequence_size)
    X_all = sequence.pad_sequences(X_all, maxlen=max_sequence_size)

    logging.info("Input shape: %s" % str(X_all.shape))
    logging.info("Output shape: %s" % str(y_all.shape))

    return perform_data_split(X_all, y_all)

def train(X_train, y_train, X_test, y_test, target_function):
    # trying with CNN model first
    model = get_cnn_model(num_amino_acids, max_sequence_size, max_num_functions, target_function)

    # moving to LSTM
    # model = get_lstm_model(num_amino_acids, max_sequence_size, max_num_functions)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # if target_function != '':
    #     loss_fn = 'categorical_crossentropy'
    # else:
    #     loss_fn = 'binary_crossentropy'

    loss_fn = 'binary_crossentropy'
    model.compile(loss=loss_fn,
                  optimizer='adam',
                  metrics=['accuracy', 'recall', 'precision', 'fbeta_score'])

    save_model_metatdata(model, batch_size, nb_epoch)

    visualization_filename = "../results/"+exp_id+"-learning.png"
    logging.info("Metrics plot file: %s" % visualization_filename)
    cb_vis_hist = LossHistory(visualization_filename, model_snapshot_directory, exp_id)
    # best_fbeta_score = 0.00

    for epoch in range(nb_epoch):
        logging.info('')
        logging.info("----------")
        logging.info("Epoch %d/%d" % (epoch+1, nb_epoch))
        logging.info("----------")
        logging.info('')
        # csv_logger = CSVLogger('training.log')
        # if file exists, load it, otherwise train


        # checkpoint = ModelCheckpoint(weights_model_filename, monitor='val_loss',
        #                                 verbose=0, save_best_only=False,
        #                                 save_weights_only=True,
        #                                 mode='auto', period=1)

        if target_function != '':
            class_weights = None # {0 : 1., 1: 100.}
        else:
            class_weights = None

        hist = model.fit(X_train, y_train,
                  batch_size = batch_size,
                  nb_epoch = 1,
                  callbacks = [cb_vis_hist],
                  validation_data = (X_test, y_test),
                  class_weight = class_weights,
                  verbose=1)

        # FIXME: best_fbeta_score = from hist
        best_fbeta_score = -0.0

        # dump these metrics as our calculation is different
        # metrics_line = 'TRAIN: '
        # for s in ['loss', 'acc', 'precision', 'recall', 'fbeta_score']:
        #     metrics_line += "%s: %.5f - " %(s, (hist.history[s])[-1])
        # logging.info(metrics_line)


        # no need to validate since we're doing it manually
        # eval_log = predict_and_eval(model, X_train, y_train, X_test, y_test)
        # cb_vis_hist.on_epoch_end(epoch, eval_log)

        # save best fbeta_score
        # new_fbeta_score = eval_log['val_fbeta_score']
        # if new_fbeta_score > best_fbeta_score:
        #     best_fbeta_score = new_fbeta_score
        #     # also save the best model
        #     model_save_filename = model_snapshot_directory + '/'+ exp_id + \
        #                                 '-saved-model.h5'
        #     model.save(model_save_filename)
        #     logging.info("-- saved best model with f measure: %.5f on epoch: %d" % (best_fbeta_score, epoch))

    return (model, best_fbeta_score)

def predict_and_eval(model, X_train, y_train, X_test, y_test, threshold = None):
    logging.info("Performing prediction on train and test for evaluation ...")
    y_pred_train = model.predict(X_train, batch_size=batch_size, verbose=1)
    y_pred_test = model.predict(X_test, batch_size=batch_size, verbose=1)

    eval_types = ['Train', 'Test']

    logs = {}

    for e, eval_type in enumerate(eval_types):
        # print "[%s]" % eval_type

        metric_prefix = '' if e == 0 else 'val_'
        X_eval = X_train if e == 0 else X_test
        y_eval = y_train if e == 0 else y_test
        y_pred_eval = y_pred_train if e == 0 else y_pred_test

        # threshold = 0.48
        # y_pred_eval = (0.5 - threshold) + y_pred_eval

        y_eval = y_eval.astype(float)

        if threshold != None:
            y_eval = K.clip((0.5 - threshold) + y_eval, 0., 1.)

        logs[metric_prefix + 'loss']        = metrics.binary_crossentropy(y_eval, y_pred_eval).eval()
        logs[metric_prefix + 'acc']         = metrics.binary_accuracy(y_eval, y_pred_eval).eval()
        logs[metric_prefix + 'precision']   = metrics.precision(y_eval, y_pred_eval).eval()
        logs[metric_prefix + 'recall']      = metrics.recall(y_eval, y_pred_eval).eval()
        logs[metric_prefix + 'fbeta_score'] = metrics.fmeasure(y_eval, y_pred_eval).eval()

        # log_file.write("%d,%.5f,%s,%.5f\n" % (epoch, threshold, eval_type, average_faux_jaccard_similarity))
        # print "%d,%.5f,%s,%.4f" % (epoch, threshold, eval_type, average_faux_jaccard_similarity)

    metrics_line = ''
    for s in ['loss', 'acc', 'precision', 'recall', 'fbeta_score']:
        metrics_line += "%s: %.5f %s: %.5f - " %(s, logs[s], 'val_'+s, logs['val_' +s])

    logging.info(metrics_line)
    return logs

def make_sample_predictions(model, X_test, y_test):
    # get some samples to predict from
    y_pred_test = model.predict(X_test, batch_size=batch_size, verbose=1)
    rnd_indices = np.random.randint(X_test.shape[0], size=samples_to_predict) # get 10 elements at random

    logging.info("Making sample predictions for function ... ")
    for i in rnd_indices:
        i = 0
        y_true = y_test[i]
        y_pred = y_pred_test[i]
        # logging.info("Predicting [True: %d | Predicted: %.4f]" %(y_true, y_pred))
        print "Predicting [True:", y_true , "| Predicted:", y_pred , "]"

        # logging.info("- Positives:")
        # for f in range(y_true.shape[0]):
        #     # first print only true ones
        #     if y_true[f] == 1:
        #         print ' ', y_true[f] , y_pred[f]
        #
        # logging.info("- Negatives:")
        # for f in range(y_true.shape[0]):
        #     if y_true[f] == 0:
        #         print ' ', y_true[f] , y_pred[f]

        # logging.info("%d %d" % (len(indices_to_functions(y_true)), indices_to_functions(y_true)))
        # logging.info("%d %d" % (len(indices_to_functions(y_pred)), indices_to_functions(y_pred)))

if __name__ == "__main__":
    # set up logging
    results_dir = "../results"
    exp_id = get_experiment_id(results_dir)
    set_logging_params(results_dir, exp_id)
    logging.info("Experiment ID: %s" % exp_id)
    print "Experiment ID: " + exp_id

    # let's also capture console output
    import sys
    oldStdout = sys.stdout
    file = open("../results/" + exp_id + "-console.txt", 'w')
    sys.stdout = file

    # --------------------------------------
    # run experiment

    if target_function != '':
        logging.debug("Target function: [" + target_function + "]")


    # sequences_file = "../data/protein-seqs-2017-01-23-203946.txt"
    # functions_file = "../data/protein-functions-2017-01-23-203946.txt"

    sequences_file = "../data/protein-seqs-2017-01-26-191058.txt"
    functions_file = "../data/protein-functions-2017-01-29-080137.txt"

    X_train, y_train, X_test, y_test = prepare_data(sequences_file=sequences_file,
                                        functions_file=functions_file,
                                        target_function=target_function)

    (model, best_fbeta_score) = train(X_train, y_train, X_test, y_test, target_function)
    # logging.debug("Got best F measure: %.5f" % best_fbeta_score)

    if samples_to_predict > 0:
        make_sample_predictions(model, X_test, y_test)


    # --------------------------------------

    # restore standard output
    sys.stdout = oldStdout
    print "Done."
