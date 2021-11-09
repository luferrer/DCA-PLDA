#!/usr/bin/env python3

# Example on how to run binary calibration with the code in the repository.

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from dca_plda import calibration
from dca_plda import scores as dscores

def generate_scores(n_tar, n_non):

    tar = np.random.randn(n_tar)
    non = np.random.randn(n_non) - 4
    scores = np.r_[tar, non]
    labels = np.zeros(scores.size)
    labels[:n_tar] = 1

    return tar, non, scores, labels
    

if __name__ == '__main__':

    np.random.seed(47)
    
    # ptar used to train the model
    ptar_train = 0.01

    # ptar used in testing
    ptar_test = 0.1

    # generate training scores
    tar, non, scores, labels = generate_scores(10000, 10000)

    a, b = calibration.logregCal(tar, non, return_params=True, Ptar=ptar_train)

    # generate testing scores
    tar, non, scores, labels = generate_scores(1000, 1000)
    tar_cal = a * tar + b
    non_cal = a * non + b
    det = dscores.Det(tar_cal, non_cal, pav=True)
    min_cllr_lin = det.min_cllr(ptar_test)
    min_cllr = det.min_cllr(ptar_test, with_pav=True)
    act_cllr = det.act_cllr(ptar_test)
    
    print("Act cllr     = %5.3f"%act_cllr)

    print("Min cllr lin = %5.3f"%min_cllr_lin)
    print("Min cllr pav = %5.3f"%min_cllr)

    print("Cal loss pav = %5.2f"%((act_cllr-min_cllr)/act_cllr*100))

    
