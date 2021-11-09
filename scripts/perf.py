import argparse
from dca_plda.scores import Scores, compute_performance

parser = argparse.ArgumentParser()
parser.add_argument('--ptar',  help='Prior for Cllr and DCF computation.', type=float, default=0.01)
parser.add_argument('--set',   help='Name for the set, to be used in the results file.', default=None)
parser.add_argument('keylist', help='List of keys for scoring.')
parser.add_argument('scores',  help='Score file in h5 format as printed by eval.py.')
parser.add_argument('outfile', help='Output file for results.')

opt = parser.parse_args()

scores = Scores.load(opt.scores)
compute_performance(scores, opt.keylist, opt.outfile, ptar=opt.ptar, setname=opt.set)

    
