import argparse
import joblib

from tp_memm import MEMM

"""
Run the transitional probability model from the command line
"""

if __name__ == "__main__":
    # load model
    lr_model = joblib.load('models/model_k5_full.pkl')
    creator = joblib.load('models/creator_k5_full.pkl')
    model = MEMM(lr_model, creator)

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--s', help="Decode a single input sentence using the logistic regression model.")
    parser.add_argument('--d', help="Decode a single input sent using the viterbi decoder.")
    args = parser.parse_args()

    if args.s:
        print(model.plain_decode(args.s + '\n'))
    if args.d:
        print(model.decode(args.d + '\n'))