import re
import glob

def dump_to_file(protein_id, sequence):
    with open(out_file, "a") as f:
        f.write(protein_id + "," + sequence + "\n")


def convert_sequences(scrape_dir, out_file, include_tr = False):
    fasta_files = glob.glob(scrape_dir + "/*.fasta")
    br = 0

    for fname in fasta_files:
        with open (fname, 'r') as f:
            # dictionary of proteins
            proteins = {}

            protein_seq = ''
            protein_id = ''

            is_tr = False

            for line in f:

                # match line to see if start of a protein descriptor
                match = re.search(r'^>([a-z]{2})\|([A-Z0-9]*)\|', line)
                if match:
                    # save old sequence
                    # proteins[protein_id] = protein_seq

                    # don't save to memory, commit to file instead
                    if protein_id != '' and (not is_tr or include_tr):
                        dump_to_file(protein_id, protein_seq)

                    protein_review_type = match.group(1)  # sp or tr
                    if protein_review_type == 'tr':
                        is_tr = True
                    else:
                        is_tr = False

                    # print "\nNew protein: ", line,
                    protein_id = match.group(2)

                    # starting new sequence
                    protein_seq = ''



                    br = br+1
                    # if br > 10: break
                else:
                    # print ".",
                    protein_seq += line.strip()

            # output the last one
            if protein_id != '':
                dump_to_file(protein_id, protein_seq)
            return


def get_target_functions(scrape_dir, cutoff_usage):
    function_counts = {}

    annot_files = glob.glob(scrape_dir + "/*annotations.txt")

    br = 0
    for fname in annot_files:
        with open (fname, 'r') as f:
            for line in f:

                match = re.search(r'([A-Z0-9]{6})\sGO:(.*);\s?F:.*?;', line)
                if match:
                    protein_id = match.group(1)
                    function = match.group(2)

                    function_counts.setdefault(function, 0)
                    function_counts[function] += 1


    # convert from dict to list of tuples
    fn_list = [(k, function_counts[k]) for k in function_counts.keys()]
    # sort according to usage
    sorted_function_counts = sorted(fn_list, key=lambda x: x[1], reverse=True)
    # remove stuff which does not have sufficient usage
    return [x[0]  for x in sorted_function_counts if x[1] >= cutoff_usage]



def convert_function(scrape_dir, out_file, dump_unique_functions=None, limit_to_functions=[]):
    annot_files = glob.glob(scrape_dir + "/*annotations.txt")

    br = 0
    proteins_functions = {}
    unique_functions = {}

    # first, get all the functions
    for fname in annot_files:
        with open (fname, 'r') as f:
            for line in f:

                # match line to see if start of a protein descriptor
                # Example: P27361 GO:0005524; F:ATP binding; IEA:UniProtKB-KW.
                match = re.search(r'([A-Z0-9]{6})\sGO:(.*);\s?F:.*?;', line)
                if match:
                    protein_id = match.group(1)
                    function = match.group(2)

                    # continue if part of not in limit_to_functions list. We're ignoring the rest
                    if function not in limit_to_functions:
                        continue

                    unique_functions[function] = True

                    # get the list of this protein from the dictionary
                    # notice the plural in proteins_function and singular in protein_function
                    protein_functions = []
                    try:
                        protein_functions = proteins_functions[protein_id]
                    except KeyError:
                        protein_functions = []
                        proteins_functions[protein_id] = protein_functions

                    protein_functions.append(function)

                br += 1
                # if br > 10: break

    # now dump to file
    import json
    with open(out_file, 'w') as fp:
        json.dump(proteins_functions, fp)

    print "Converted %d protein functions: " % len(proteins_functions)

    if dump_unique_functions != None:
        keys = list(unique_functions.keys())
        with open (dump_unique_functions, 'wb') as fp:
            for k in keys:
              fp.write("%s\n" % k)


if __name__ == '__main__':
    import datetime, time
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H%M%S')

    scrape_dir = "../data-scrapes"

    # convert sequences
    print "Converting sequences ... "
    out_file = "../data/protein-seqs-" + st + ".txt"
    convert_sequences(scrape_dir, out_file)

    # convert function
    print "Converting functions ..."
    out_file_fns = "../data/protein-functions-" + st + ".txt"
    out_file_unique_functions = "../data/unique_functions_cutoff_1600.txt"

    # keep only those functions which are used at least cutoff times
    function_usage_cutoff = 1600
    target_functions = get_target_functions(scrape_dir, function_usage_cutoff)
    print "Number of target functions: ", len(target_functions)
    convert_function(scrape_dir, out_file_fns, dump_unique_functions=out_file_unique_functions, limit_to_functions=target_functions)
