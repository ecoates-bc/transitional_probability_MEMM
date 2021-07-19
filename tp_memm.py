import numpy as np
import sklearn
import joblib

import feature_creator

"""
A Maximum-Entropy Markov Model for predicting word boundaries,
using a feature space of transitional probabilities
"""


class MEMM:
    def __init__(self, preload_model=None, preload_creator=None):
        if preload_model:
            self.model = preload_model
        else:
            self.model = sklearn.linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')

        if preload_creator:
            self.creator = preload_creator
        else:
            self.creator = feature_creator.FeatureCreator('data/train.src', 'data/train.trg')

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, x):
        try:
            return self.model.predict(x)
        except ValueError:
            return self.model.predict(x.reshape(1, -1))

    def get_weights(self):
        return self.model.coef_

    def entropy(self, f):
        return self.model.predict_proba(f.reshape(1, -1))[0]

    def decode(self, line, features=None):
        # find word boundaries using the Viterbi algorithm
        # line is a text line

        T = len(line)
        F = features if features is not None else self.creator.get_feature_matrix(line)
        viterbi = np.zeros((T, 2))
        backpointer = np.zeros((T-1, 2))

        for t in range(T):
            if t == 0:
                viterbi[t] = self.entropy(F[t])
            else:
                v_0 = viterbi[t-1, 0] * self.entropy(F[t])
                viterbi[t, 0] = np.max(v_0)
                backpointer[t-1, 0] = np.argmax(v_0)

                v_1 = viterbi[t-1, 1] * self.entropy(F[t])
                viterbi[t, 1] = np.max(v_1)
                backpointer[t-1, 1] = np.argmax(v_1)

        # follow the backpointers to get the best sequence
        max_end = np.argmax(viterbi[-1])
        seq = np.array([max_end])
        for i in reversed(range(T-2)):
            prev_idx = np.array([backpointer[i, int(seq[0])]])
            seq = np.concatenate((prev_idx, seq))

        seq = np.concatenate((np.array([1]), seq))

        # convert the text and the backtracked sequence into segmented text
        return self._convert_to_segmented(line, seq)

    def plain_decode(self, line, features=None):
        # decode a line without the viterbi algorithm
        F = features if features is not None else self.creator.get_feature_matrix(line)
        y = np.array([self.predict(F[i]) for i in range(F.shape[0])])

        return self._convert_to_segmented(line, y)

    def _convert_to_segmented(self, line, idx):
        # convert an unsegmented line and a vector of indices into segmented text
        segm_line = []
        for i in range(len(line)-1):
            if idx[i]:
                segm_line += '.' + line[i]
            else:
                segm_line += line[i]
        return ''.join(segm_line)[1:]


if __name__ == "__main__":
    model = MEMM()

    # read features saved to disk
    X = np.genfromtxt('transformed_data/train.src', delimiter='\t')
    y = np.genfromtxt('transformed_data/train.trg', delimiter='\t')
    n, d = X.shape

    X_val = np.genfromtxt('transformed_data/dev.src', delimiter='\t')
    y_val = np.genfromtxt('transformed_data/dev.trg', delimiter='\t')
    n_val, _ = X_val.shape

    print('Training model...')
    model.fit(X, y)

    # calculate training/validation errors without viterbi decoding
    y_tilde = model.predict(X)
    y_hat = model.predict(X_val)

    err_train = np.average([y[i] != y_tilde[i] for i in range(n)])
    print('Training error: {}'.format(err_train * 100))

    err_val = np.average([y_val[i] != y_hat[i] for i in range(n_val)])
    print('Validation error: {}'.format(err_val * 100))

    # try out decoding (last 50 lines of the training set)
    line_orig = [line for line in open('data/train.src', 'r').readlines()][-50:]
    line_gold = [line for line in open('data/train.trg', 'r').readlines()][-50:]
    line_predict = [model.plain_decode(line) for line in line_orig]

    with open('dev_decode.txt', 'w') as out_file:
        for i in range(len(line_predict)):
            out_file.write('SYS: ' + line_predict[i] + '\n')
            out_file.write('GOLD: ' + line_gold[i])
            out_file.write('\n')

    # save model
    joblib.dump(model.model, 'models/model_k5_full.pkl')
    joblib.dump(model.creator, 'models/creator_k5_full.pkl')