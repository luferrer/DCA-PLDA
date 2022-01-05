import argparse
import os
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

from dca_plda.data import LabelledDataset
from dca_plda.utils_for_scripts import np_to_torch, evaluate, load_model, mkdirp, setup_torch_and_cuda_vars
from dca_plda.scores import IdMap, compute_performance

parser = argparse.ArgumentParser()
parser.add_argument('--cuda',     help='Enables cuda.', action='store_true')
parser.add_argument('--durs',     help='File with durations, needed only if the model was created with duration_dependent_calibration = True.', default=None)
parser.add_argument('--min_dur',  help="Minimum duration in seconds below which all scores for a model or test are turned to 0.", default=0, type=float)
parser.add_argument('--keylist',  help='List of keys for scoring. If not provided, only the scores are printed in the out_dir.', default=None)
parser.add_argument('--ptar',     help='Prior for Cllr and DCF computation if keylist is provided.', default=0.01, type=float)
parser.add_argument('--set',      help='Name for the set, to be used in the results file.', default=None)
parser.add_argument('--raw',      help='Output the raw scores, before the calibration stages (used for analysis).', default=False, action='store_true')
parser.add_argument('--level',    help='Which LLRs to return for hierarchical DCA-PLDA: None (LLRs for the classes), level1 (LLRs for clusters), level2 (LLRs for clusters conditioned on known cluster).', default=None)
parser.add_argument('--cluster_priors', help='Only for hierarchical DCA-PLDA. A table with the detection prior for each cluster. If provided, the output LLRs will be computed using these priors instead of the default ones.', action=None)
parser.add_argument('--fmt',      help='Format of output scores: h5 or ascii.', default='h5')
parser.add_argument('model',      help='Path to the model to be used for evaluation.')
parser.add_argument('embeddings', help='Path to the npz file with development embeddings.')
parser.add_argument('enroll_map', help='Map from enrollment ids (first column) to the ids used in the embeddings file (second column).\
                                        The mapping could be one to many for multi-session enrollment.')
parser.add_argument('test_map',   help='Map from test ids (first column) to the ids used in the embeddings file (second column).')
parser.add_argument('out_dir',    help='Output directory for scores and results, if keylist is provided.')

opt = parser.parse_args()
mkdirp(opt.out_dir)

##### Set the device and data type
device = setup_torch_and_cuda_vars(opt.cuda)

###### Load the model
model = load_model(opt.model, device)
print("Loaded model from %s"%opt.model)

###### Load the data
dataset = LabelledDataset(opt.embeddings, opt.durs, meta_is_dur_only=True, device=device, skip_missing=True)
if model.enrollment_classes is not None:
    assert opt.enroll_map == 'NONE'
    if opt.level == 'level1':
        emap = IdMap.load('NONE', model.level1_detector.enrollment_classes)
    else:
        emap = IdMap.load('NONE', model.enrollment_classes)
    eids = emap.model_ids
else:
    emap = IdMap.load(opt.enroll_map, dataset.get_ids())
    eids = None

tmap = IdMap.load(opt.test_map, dataset.get_ids())
    
###### Generate the scores
cluster_prior_dict = dict(np.loadtxt(opt.cluster_priors, dtype='O', converters={1: float})) if opt.cluster_priors else None
scores = evaluate(model, dataset, emap, tmap, min_dur=opt.min_dur, raw=opt.raw, level=opt.level, cluster_prior_dict=cluster_prior_dict)
scores.save("%s/scores.%s"%(opt.out_dir, opt.fmt), fmt=opt.fmt)

if opt.keylist is not None:
    compute_performance(scores, opt.keylist, "%s/results"%opt.out_dir, ptar=opt.ptar, setname=opt.set, enrollment_ids=eids)
    
