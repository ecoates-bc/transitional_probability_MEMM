import joblib
from nltk import edit_distance

from tp_memm import MEMM

"""
Calculate test accuracy, using line-level recall and average edit distance
"""


def line_acc(sys, gold):
    if sys == gold:
        return 1
    else:
        return 0


def avg_line_acc(sys_list, gold_list):
    assert len(sys_list) == len(gold_list)
    return sum([line_acc(sys_list[i], gold_list[i]) for i in range(len(sys_list))]) / len(sys_list)


def avg_edit_dist(sys_list, gold_list):
    assert len(sys_list) == len(gold_list)
    return sum([edit_distance(sys_list[i], gold_list[i]) for i in range(len(sys_list))]) / len(sys_list)


def avg_normalized_edit_dist(sys_list, gold_list):
    assert len(sys_list) == len(gold_list)
    return sum([edit_distance(sys_list[i], gold_list[i]) / len(sys_list[i])
                for i in range(len(sys_list))]) / len(sys_list)


if __name__ == "__main__":
    # load test data
    test_src = [line for line in open('data/test.src').readlines()]
    test_trg = [line.rstrip() for line in open('data/test.trg').readlines()]

    # load model
    lr_model = joblib.load('models/model_k5_full.pkl')
    creator = joblib.load('models/creator_k5_full.pkl')
    model = MEMM(lr_model, creator)

    # pre-compute features
    features = []
    for i in range(len(test_src)):
        features.append(model.creator.get_feature_matrix(test_src[i]))
        print('{} of {}'.format(i+1, len(test_src)))

    # predict test labels
    viterbi_sys = [model.decode(test_src[i], features[i]) for i in range(len(test_src))]
    logreg_sys = [model.plain_decode(test_src[i], features[i]) for i in range(len(test_src))]

    # calculate, print the metrics
    print('Plain old logistic regression:')
    print('Line-based recall: {:.2f}'.format(avg_line_acc(logreg_sys, test_trg)))
    print('Average edit distance: {:.2f}'.format(avg_edit_dist(logreg_sys, test_trg)))
    print('Average normalized edit distance: {:.2f}'.format(avg_normalized_edit_dist(logreg_sys, test_trg)))

    print()

    print('Maximum-Entropy Markov decoding:')
    print('Line-based recall: {:.2f}'.format(avg_line_acc(viterbi_sys, test_trg)))
    print('Average edit distance: {:.2f}'.format(avg_edit_dist(viterbi_sys, test_trg)))
    print('Average normalized edit distance: {:.2f}'.format(avg_normalized_edit_dist(viterbi_sys, test_trg)))

    # save predictions
    with open('test_decode.txt', 'w') as out_file:
        for i in range(len(logreg_sys)):
            out_file.write('SYS: ' + logreg_sys[i] + '\n')
            out_file.write('GOLD: ' + test_trg[i] + '\n')
            out_file.write('\n')

    with open('test_viterbi_decode.txt', 'w') as out_file:
        for i in range(len(viterbi_sys)):
            out_file.write('SYS: ' + viterbi_sys[i] + '\n')
            out_file.write('GOLD: ' + test_trg[i] + '\n')
            out_file.write('\n')