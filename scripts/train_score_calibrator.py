#!/usr/bin/env python3

import argparse
from dca_plda import calibration
from dca_plda import scores as dscores
import numpy as np

parser = argparse.ArgumentParser(description="Train a post-hoc score calibrator using linear logistic regression.")
parser.add_argument('--key',      help='Key used for calibration. Required when scores are in h5 format.', default=None)
parser.add_argument('--ptar',     help='Effective prior for positive samples used during training. This parameter should coincide roughly (ie, in order of magnitude) to the effective prior that will be used to evaluate the system but it usually does not need to be identical.', default=0.5, type=float)
parser.add_argument('--fmt',      help='Format of input scores. If set to "npz", an npz file with two arrays, pos and neg, for the positive and negative samples is expected.', default='npz')
parser.add_argument('scores',     help='Score file in h5 or npz format.')
parser.add_argument('model',      help='File where to print the calibration parameters, alpha (scale) and beta (shift).')

opt = parser.parse_args()


if opt.fmt == "h5":
    assert opt.key != None
    scores = dscores.Scores.load(opt.scores)
    key = dscores.Key.load(opt.key)
    ascores = scores.align(key)
    pos = ascores.score_mat[key.mask==1]
    neg = ascores.score_mat[key.mask==-1]
                                                                    
else:
    sc = np.load(opt.scores)
    pos, neg = sc['pos'], sc['neg'] 

a, b = calibration.logregCal(pos, neg, return_params=True, Ptar=opt.ptar)

np.savez(opt.model, a=a, b=b)

print("Trained linear calibration model with %d positive and %d negative samples."%(len(pos), len(neg)))
print("Alpha = %.4f"%a)
print("Beta  = %.4f"%b)
