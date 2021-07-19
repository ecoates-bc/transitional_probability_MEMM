import itertools
import numpy as np
from nltk import bigrams
from sklearn.preprocessing import KBinsDiscretizer

"""
Transform a character into a feature-vector based on unigram/bigram TP.
"""


class FeatureCreator:
    def __init__(self, src_path, trg_path):
        # load training text
        print('Loading data...')
        self.src = [line.rstrip() for line in open(src_path, 'r').readlines()]
        self.trg = [line.rstrip().split('.') for line in open(trg_path, 'r').readlines()]

        # create unigram/bigram inventories
        print('Creating inventories...')
        self.chars = sorted(list(set(''.join(self.src))))
        self.bigrams = sorted(self._create_bigrams())

        # create frequency matrices for calculating TP
        print('Generating frequencies...')
        self.unigram_freq_mat = self._create_unigram_freq_matrix()
        self.bigram_freq_mat = self._create_bigram_freq_matrix()

        # create discretizers for creating feature-vectors
        print('Fitting discretizers...')
        self.n_bins = 5
        self.unigram_discretizer = self._get_unigram_discretizer()
        self.bigram_discretizer = self._get_bigram_discretizer()

    def _create_unigram_freq_matrix(self):
        # return a matrix of word-internal co-occurrence frequencies for unigrams

        mat = np.zeros((len(self.chars), len(self.chars)))
        for line in self.trg:
            for word in line:
                for i in range(len(word)):
                    if i > 0:
                        mat[self.chars.index(word[i - 1])][self.chars.index(word[i])] += 1
        return mat

    def _create_bigrams(self):
        grams = set()
        for line in self.trg:
            for word in line:
                for b in bigrams(word):
                    grams.add(''.join(b))
        return list(grams)

    def _create_bigram_freq_matrix(self):
        # find frequencies of word-internal bigrams

        mat = np.zeros((len(self.bigrams), len(self.bigrams)))
        for line in self.trg:
            for word in line:
                grams = [''.join(b) for b in bigrams(word)]
                for i in range(len(grams)):
                    if i > 1:
                        mat[self.bigrams.index(grams[i - 2])][self.bigrams.index(grams[i])] += 1
        return mat

    def get_tp(self, a, b, gram_type, dir):
        # check if we're calculating unigram or bigram TP
        if gram_type == 'unigram':
            assert len(a) == len(b) == 1
            freq_mat = self.unigram_freq_mat
            segment_list = self.chars
        elif gram_type == 'bigram':
            assert len(a) == len(b) == 2
            freq_mat = self.bigram_freq_mat
            segment_list = self.bigrams
        else:
            raise ValueError('gram_type must be \'unigram\' or \'bigram.\'')

        # find the frequencies of each segment from the matrix
        try:
            n_bigram = freq_mat[segment_list.index(a)][segment_list.index(b)]
        except ValueError:
            raise ZeroDivisionError('A segment is not found in the corpus.')
        if n_bigram == 0:
            raise ZeroDivisionError('The bigram ({}, {}) is not found in the corpus.'.format(a, b))
        total_bigrams = np.sum(np.sum(freq_mat))
        n_a_bigrams = np.sum(freq_mat[segment_list.index(a)])
        n_b_bigrams = np.sum(freq_mat[:, segment_list.index(b)])

        # find individual probabilities
        p_bigram = n_bigram / total_bigrams
        p_a = n_a_bigrams / total_bigrams
        p_b = n_b_bigrams / total_bigrams

        if dir == 'f':
            return p_bigram / p_a
        elif dir == 'b':
            return p_bigram / p_b
        else:
            raise ValueError('TP direction must be \'f\' or \'b.\'')

    def _get_unigram_discretizer(self):
        discr = KBinsDiscretizer(n_bins=self.n_bins, strategy='quantile')

        tps = []
        for i, j in itertools.product(self.chars, self.chars):
            try:
                tps.append(self.get_tp(i, j, 'unigram', 'f'))
            except ZeroDivisionError:
                pass
        discr.fit(np.array(tps).reshape(-1, 1))
        return discr

    def _get_bigram_discretizer(self):
        discr = KBinsDiscretizer(n_bins=self.n_bins, strategy='quantile')
        tps = []
        for i, j in itertools.product(self.bigrams, self.bigrams):
            try:
                tps.append(self.get_tp(i, j, 'bigram', 'f'))
            except ZeroDivisionError:
                pass
        discr.fit(np.array(tps).reshape(-1, 1))
        return discr

    def _get_discrete_tp(self, a, b, seg_type, dir):
        # return a one-hot vector from the unigram TP discretizer
        try:
            tp = np.array(self.get_tp(a, b, seg_type, dir)).reshape(1, -1)
            if seg_type == "unigram":
                transform = self.unigram_discretizer.transform(tp).toarray()[0]
            else:
                transform = self.bigram_discretizer.transform(tp).toarray()[0]
            return transform
        except ZeroDivisionError:
            return np.zeros(self.n_bins)

    def _get_character_feature_vector(self, char, prev_unigram, next_unigram, bigram, prev_bigram, next_bigram):
        # get forward and backward unigram TP
        unigram_prev_f = self._get_discrete_tp(prev_unigram, char, 'unigram', 'f')
        unigram_prev_b = self._get_discrete_tp(prev_unigram, char, 'unigram', 'b')
        unigram_next_f = self._get_discrete_tp(char, next_unigram, 'unigram', 'f')
        unigram_next_b = self._get_discrete_tp(char, next_unigram, 'unigram', 'b')

        # get forward and backward bigram TP
        bigram_prev_f = self._get_discrete_tp(prev_bigram, bigram, 'bigram', 'f')
        bigram_prev_b = self._get_discrete_tp(prev_bigram, bigram, 'bigram', 'b')
        bigram_next_f = self._get_discrete_tp(bigram, next_bigram, 'bigram', 'f')
        bigram_next_b = self._get_discrete_tp(bigram, next_bigram, 'bigram', 'b')

        return np.concatenate((
            unigram_prev_f, unigram_prev_b, unigram_next_f, unigram_next_b,
            bigram_prev_f, bigram_prev_b, bigram_next_f, bigram_next_b
        ))

    def get_feature_matrix(self, line):
        # get feature vectors for each character in a src line
        padded_line = '##%s###' % line

        # unigram, prev. unigram, next unigram,
        # bigram, prev. bigram, next bigram
        sent_mat = np.array([self._get_character_feature_vector(padded_line[i], padded_line[i - 1], padded_line[i + 1],
                                                                padded_line[i:i + 2], padded_line[i - 2:i],
                                                                padded_line[i + 2:i + 4])
                             for i in range(2, len(line) + 2)])
        return sent_mat

    def get_data_label_pair(self, line_idx):
        # create an X matrix and y vector from a training example
        src = self.src[line_idx]
        trg = self.trg[line_idx]

        # create labels
        labels = []
        for w in trg:
            for i in range(len(w)):
                if i == 0:
                    # start character, word boundary precedes it
                    labels.append(1)
                else:
                    labels.append(0)

        y = np.asarray(labels)

        # create data
        X = self.get_feature_matrix(src)

        assert X.shape[0] == y.shape[0]
        return X, y

    def transform_dataset(self, data_path, label_path, val_pct=0., val_data_path=None, val_label_path=None):
        # Transform all data in the training set and save
        print('Transforming dataset...')
        X, y = self.get_data_label_pair(0)

        for i in range(1, len(self.src)):
            next_x, next_y = self.get_data_label_pair(i)
            X = np.concatenate((X, next_x))
            y = np.concatenate((y, next_y))
            print('{} of {}...'.format(i + 1, len(self.src)))

        # create validation set
        if val_pct > 0:
            n, d = X.shape
            n_val = int(n * val_pct)
            X_val, y_val = X[n - n_val:], y[n - n_val:]
            X, y = X[:n - n_val], y[:n - n_val]

        # save to disk
        np.savetxt(data_path, X, fmt='%d', delimiter='\t')
        np.savetxt(label_path, y, fmt='%d', delimiter='\t')

        if val_data_path and val_label_path:
            np.savetxt(val_data_path, X_val, fmt="%d", delimiter='\t')
            np.savetxt(val_label_path, y_val, fmt='%d', delimiter='\t')


if __name__ == "__main__":
    creator = FeatureCreator('data/train.src', 'data/train.trg')
    creator.transform_dataset('transformed_data/train.src', 'transformed_data/train.trg', 0.1,
                              'transformed_data/dev.src', 'transformed_data/dev.trg')
