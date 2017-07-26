import numpy as np
import glob
import re
import logging

from keras.preprocessing import sequence

def get_protein_sequence(sequences_file, target_protein_id, max_sequence_size):
    seq = None
    with open(sequences_file) as f:
        for line in f:
            ln = line.split(',')
            protein_id = ln[0].strip()

            if protein_id == target_protein_id:
                seq = ln[1].strip()
                break

    if seq == None:
        return None

    original_size = len(seq)
    X = [ sequence_to_indices(seq) ]
    X = np.array(X)
    X = sequence.pad_sequences(X, maxlen=max_sequence_size)[0]
    X = X.reshape(1, max_sequence_size)
    return (X, original_size)


def get_current_timestamp():
    import datetime, time
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H%M%S')
    return st

acid_letters = ['_', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
                'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

def sequence_to_indices(sequence):
    try:
        indices = [acid_letters.index(c) for c in list(sequence)]
        return indices
    except Exception:
        print sequence
        raise Exception

def indices_to_sequence(indices):
    return ''.join([acid_letters[i] for i in indices if i != 0])

def one_hot_amino_acids(sequence, out_seq_size=None):
    # convert sequence to indices
    seq_indices = sequence_to_indices(sequence)
    seq_size = len(seq_indices)

    num_possible_amino_acids = 22 # hard coded

    # Create zeros and set appropriate 1s
    one_hot_vec = np.zeros((seq_size, num_possible_amino_acids), dtype=np.int)
    one_hot_vec[np.arange(seq_size), seq_indices] = 1


    # if don't need padding
    if out_seq_size == None:
       return one_hot_vec

    # need padding
    pad_array = np.zeros((out_seq_size - one_hot_vec.shape[0], num_possible_amino_acids), dtype=one_hot_vec.dtype)
    return np.concatenate((one_hot_vec, pad_array))


def one_hot_to_seq_amino_acids(one_hot_vec):
    # find non zero indices, convert to list, lookup in amino acid letters and join
    indices = list(one_hot_vec.nonzero()[1]) # nonzero takes care of padding automatically
    return indices_to_sequence(indices)


## example run for one hot to-and-from
# seq = 'VDACG'
# one_hot_vec = one_hot_amino_acids(seq, out_seq_size=6)
# seq_back = one_hot_to_seq_amino_acids(one_hot_vec)
# print one_hot_vec, seq, seq_back
#
# print sequence_to_indices(seq)
# print indices_to_sequence([0, 1, 2, 3])


def get_possible_functions():
    # unique_function_file = '../data/unique_functions_cutoff_10.txt'
    unique_function_file = '../data/unique_functions_cutoff_1600.txt'

    functions = []
    with open(unique_function_file, 'r') as f:
        functions = f.readlines()

    functions = [l.strip() for l in functions]
    return functions

def functions_to_indices(functions):
    possible_functions = get_possible_functions()
    fn_array_size = len(possible_functions)

    indices = np.zeros(fn_array_size, dtype=np.int)
    for fn in functions:
        indices[possible_functions.index(fn)] = 1

    return np.array(indices)

def indices_to_functions(indices):
    possible_functions = get_possible_functions()
    return [possible_functions[i] for i in indices.nonzero()[0]]


# f = functions_to_indices([['0070401', '0044610'],
#                           ['0070401']])
# print f.shape
# print indices_to_functions(f)


def faux_jaccard_simlarity(a, b):
    return len(set(a).intersection(set(b))) / float(len(set(a).union(set(b))))


# some logging stuff
def get_experiment_id(results_dir):
    ts = get_current_timestamp()
    all_files = glob.glob(results_dir + '/**.txt')
    all_files

    max_id = 0
    for f in all_files:

        match = re.search(r'([0-9]{5})', f)
        if match:
            id = int(match.group(1))
            max_id = id if id > max_id else max_id

    exp_id = "%05d" % (max_id+1)
    return exp_id


def set_logging_params(results_dir, exp_id):
    log_file = results_dir + "/" + exp_id + "-results.txt"
    logging.basicConfig(format='%(asctime)s [%(levelname)7s] %(message)s', filename=log_file, level=logging.DEBUG)


def save_model_metatdata(model, batch_size, epochs):
    logging.debug("Batch size = " + str(batch_size))
    logging.debug("Epochs = " + str(epochs))
    logging.debug(model.to_json())
