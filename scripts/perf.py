import argparse
from dca_plda.scores import Scores, compute_performance, compute_performance_with_confidence_intervals

parser = argparse.ArgumentParser()
parser.add_argument('--ptar',  help='Prior for Cllr and DCF computation.', type=float, default=0.01)
parser.add_argument('--set',   help='Name for the set, to be used in the results file.', default=None)
parser.add_argument('--conf',  help='Version of results with confidence intervals for EER, Actual and Min DCF, and Actual and Min Cllr for the provided ptar. Requires --spk and --sess.', action='store_true')
parser.add_argument('--spk',   help='Table with test and enrollment ids mapped to speaker name. Used only when --conf is set.', default=None)
parser.add_argument('--ses',   help='Table with test and enrollment ids mapped to session name. Used only when --conf is set.', default=None)
parser.add_argument('--nboot', help='Number of bootstrap samples to use for computing confidence intervals.', type=int, default=100)
parser.add_argument('keylist', help='List of keys for scoring.')
parser.add_argument('scores',  help='Score file in h5 format as printed by eval.py.')
parser.add_argument('outfile', help='Output file for results.')

opt = parser.parse_args()

scores = Scores.load(opt.scores)
if opt.conf:
    spk_dict = dict([l.strip().split() for l in open(opt.spk).readlines()])
    ses_dict = dict([l.strip().split() for l in open(opt.ses).readlines()]) if opt.ses else None
    compute_performance_with_confidence_intervals(scores, opt.keylist, opt.outfile, spk_dict, ses_dict, ptar=opt.ptar, setname=opt.set, num_boot_samples=opt.nboot, percentile=5)
else:
    compute_performance(scores, opt.keylist, opt.outfile, ptar=opt.ptar, setname=opt.set)

    
