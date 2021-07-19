import random

from nltk.corpus import gutenberg
from nltk import download
import re
from random import shuffle

# download('gutenberg')

austen = ['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt']
austen_sents = [[s for s in gutenberg.sents(i)] for i in austen]

train_dset = austen_sents[1] + austen_sents[2]  # the two shortest books
test_dset = austen_sents[1]


def create_formatted_line(line):
    # take in a segmented sentence and return two pairs:
    # the source sentence that is all lower-case and no word boundaries,
    # and the target sentence that is all lower-case and has periods marking word boundaries.
    # the input sentence has punctuation cleaned, and is more than one token long

    lc_sentence = [w.lower() for w in line]
    src = ''.join(lc_sentence)
    trg = '.'.join(lc_sentence)
    return src, trg


def format_dataset(dset):
    # creates a dataset of src,trg pairs.
    # will call create_formatted_line if the non-punct line is longer than one token
    # returns a shuffled list of pairs

    cleaned_dset = [[t for t in i if re.search(r'[a-zA-Z]', t)] for i in dset]
    dset_pairs = [create_formatted_line(i) for i in cleaned_dset if 5 < len(i) < 20]
    shuffle(dset_pairs)

    return dset_pairs


train_dset = format_dataset(train_dset)
test_dset = format_dataset(test_dset)


def create_files(dset, filename):
    with open(filename + '.src', 'w') as src_file:
        for line in dset:
            src_file.write(line[0] + '\n')
    with open(filename + '.trg', 'w') as trg_file:
        for line in dset:
            trg_file.write(line[1] + '\n')


create_files(train_dset, 'train')
create_files(test_dset, 'test')