import numpy as np
np.random.seed(1337)  # for reproducibility, needs to be the first 2 lines in a script

import logging

from keras.models import Model, load_model
from train import prepare_data, predict_and_eval
from keras import metrics
from utils import *
import json

max_sequence_size = 2000

batch_size = 500

functions_map = ['0046872', '0003676', '0008270', '0003677', '0005524']

def get_sequences_with_unkonwn_annotations(sequences_file, functions_file):
    # input protein-function map
    protein_function_map = {}
    with open(functions_file) as fn_file:
        protein_function_map = json.load(fn_file)

    proteins_of_interest = []
    # open sequences file. Has:  protein_id,seq
    i = 0
    with open(sequences_file) as f:
        for line in f:
            ln = line.split(',')
            protein_id = ln[0].strip()
            seq = ln[1].strip()

            if protein_id in protein_function_map:
                continue # have annotation. Don't need it.

            seq_x = sequence_to_indices(seq)
            seq_x = sequence.pad_sequences([seq_x], maxlen=max_sequence_size)
            seq_x = seq_x[0]
            proteins_of_interest.append((protein_id, seq, seq_x))

            i += 1
            # if i > 2: break

    return proteins_of_interest


def predict(model, proteins_of_interest, out_file):

    # get sequences only
    seqs = np.array([i[2] for i in proteins_of_interest])
    predictions = model.predict(seqs, verbose=1)

    out_str = ''
    for protein_index, pred in enumerate(predictions):
        for i, val in enumerate(pred):
            if val >= 0.5:
                out_str += proteins_of_interest[protein_index][0] + ' GO:' + functions_map[i] + ";\n"


    with open(out_file, "w") as text_file:
        text_file.write(out_str)

if __name__ == '__main__':
    model_filename = "../results/00061-saved-model.h5"
    sequences_file = "../data/protein-seqs-2017-01-23-203946.txt"
    functions_file = "../data/protein-functions-2017-01-23-203946.txt"
    out_file = "../data/predictions-unknowns-00002"


    proteins_of_interest = get_sequences_with_unkonwn_annotations(sequences_file, functions_file)
    # print seqs

    model = load_model(model_filename)
    predict(model, proteins_of_interest, out_file)
