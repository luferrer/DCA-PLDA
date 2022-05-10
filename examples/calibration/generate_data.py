#!/usr/bin/env python3

# Example on how to run binary calibration with the code in the repository.

import numpy as np

def generate_scores(n_tar, n_non):

    tar = np.random.randn(n_tar)
    non = np.random.randn(n_non) - 4
    scores = np.r_[tar, non]
    labels = np.zeros(scores.size)
    labels[:n_tar] = 1

    return tar, non, scores, labels
    

if __name__ == '__main__':

    np.random.seed(47)
    
    # generate training scores
    pos, neg, scores, labels = generate_scores(10000, 100000)
    np.savez("trn_scores.npz", pos=pos, neg=neg)

    # generate testing scores
    pos, neg, scores, labels = generate_scores(1000, 10000)
    np.savez("tst_scores.npz", pos=pos, neg=neg)
    np.savez("tst_scores_nokey.npz", all=np.r_[pos,neg])

    
