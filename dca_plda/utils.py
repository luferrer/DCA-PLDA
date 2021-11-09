# This file contains utilities that do not need imports from the repository

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import sklearn.metrics
import configparser
import glob
import re
import shutil
import torch.optim as optim
import os
from numpy.linalg import cholesky as chol
from scipy.linalg import solve_triangular
from pathlib import Path


def save_checkpoint(model, outdir, epoch=None, trn_loss=None, dev_loss=None, optimizer=None, scheduler=None, name="model"):
    
    if trn_loss is not None:
        lr = optimizer.param_groups[0]['lr']
        outfile = '%s/%s_epoch_%04d_lr_%8.6f_trnloss_%.4f_devloss_%.4f.pth' % (outdir, name, epoch, lr, trn_loss, dev_loss)
    else:   
        if dev_loss is not None:
            if epoch is not None:
                outfile = '%s/%s_epoch_%04d_devloss_%.4f.pth' % (outdir, name, epoch, dev_loss)
            else:
                outfile = '%s/%s_devloss_%.4f.pth' % (outdir, name, dev_loss)
        else:
            if epoch is not None:
                outfile = '%s/%s_epoch_%04d.pth' % (outdir, name, epoch)
            else:
                outfile = '%s/%s.pth' % (outdir, name)

    print("Saving model in %s"%outfile)
    save_dict = {'model': model.state_dict(), 
                 'in_dim': model.in_dim,
                 'config': model.config,
                 'optimizer': optimizer.state_dict() if optimizer is not None else None,
                 'scheduler': scheduler.state_dict() if scheduler is not None else None}
    torch.save(save_dict, outfile)

    return outfile


def load_checkpoint(file, model, device, optimizer=None, scheduler=None):

    loaded_dict = torch.load(file, map_location=device)
    model.load_state_dict(loaded_dict['model'])
    if optimizer:
        optimizer.load_state_dict(loaded_dict['optimizer'])
    if scheduler:
        scheduler.load_state_dict(loaded_dict['scheduler'])


def update_model_with_weighted_average(model, checkpoint, device, n):

    new_params = torch.load(checkpoint, map_location=device)['model']
    prev_params = model.state_dict()
    upd_params = dict()
    for p in prev_params.keys():
        upd_params[p] = (prev_params[p]*n + new_params[p])/(n+1)
    model.load_state_dict(upd_params)

def replace_state_dict(model, sdict):

    state_dict = model.state_dict()
    for k, v in sdict.items():
        state_dict[k] = torch.tensor(v)
    model.load_state_dict(state_dict)


def create_scoring_masks(metadata, class_name='class_id'):

    def _same_val_mat(f):
        ids = metadata[f].unsqueeze(1)
        return ids == ids.T
    
    same_cla = _same_val_mat(class_name)
    same_dom = _same_val_mat('domain_id')
    same_ses = _same_val_mat('session_id')
    # Valid trials are those with same domain and different session. The other trials are not scored.
    # While the batches are created so that all target trials come from different sessions, some of the
    # impostor trials could be same-session in some datasets (ie, when a session involves two sides of
    # a phone conversation). We need to discard those trials here.
    valid = same_dom * ~same_ses

    return same_cla, valid


def load_config(configfile, default_config):

    config_out = dict()
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())

    if not os.path.isfile(configfile):
        raise Exception("Config file %s does not exist"%configfile)

    try:
        config.read(configfile)
    except Exception as inst:
        print(inst)
        raise Exception("Error parsing config file %s"%configfile)

    # Use eval to conveniently convert the strings to their desired type
    # While this would be dangerous in general, I am assuming this code will only be used in safe environments.
    for sec in list(default_config.keys())+config.sections():
        config_out[sec] = dict(default_config[sec]) if sec in default_config else dict()
        if sec in config:
            for k, v in config[sec].items():
                config_out[sec][k] = eval(v)
        config_out[sec] = AttrDict(config_out[sec])

    return AttrDict(config_out)

def map_to_consecutive_ids(in_ids):
    # Map a set of ids with gaps (eg, [0, 0, 3, 2, 3], ie, the 1 is missing)
    # to a set if ids without gaps ([0, 0, 2, 1, 2])
    orig_ids, out_ids = np.unique(in_ids, return_inverse=True)
    id_map = dict([o, i] for i, o in enumerate(orig_ids))
    return id_map, out_ids

def combine_configs(config_dict, default_config_dict):

    config_out = default_config_dict.copy()
    for k, v in config_dict.items():
        config_out[k] = v
    return AttrDict(config_out)


def load_configs(configfiles, default_config, mods, outfile):

    config = default_config
    for configfile in configfiles.split(','):
        config = load_config(configfile, config)

    # Finally, after reading all the config files, override the parameters using the mods
    if mods is not None:
        for pair in mods.split(","):
            n, v = pair.split("=")
            sec, param = n.split(".")
            config[sec][param] = eval(v)

    # Write the resulting config to file and print it to stdout
    print_config(config, outfile)

    return config


def assign_default_config_values(config, default_config, outfile):

    merged_config = default_config.copy()
    for k, v in config.items():
        merged_config[k] = config[k]
    print_config(merged_config, outfile, has_sections=False)

    return AttrDict(merged_config)


def print_config(config, outfile, has_sections=True):

    outf = open(outfile, "w")
    if has_sections:
        for sec, d in config.items():
            outf.write("[%s]\n"%sec)
            for k, v in d.items():
                outf.write("  %s = %s\n"%(k,str(v)))
    else:
        for k, v in config.items():
            outf.write("%s = %s\n"%(k,str(v)))
    outf.close()

    print("Configuration parameters:")
    with open(outfile, 'r') as fin:
        for l in fin.readlines():
            print("  "+l.rstrip())


def find_checkpoint(dir, which="last", n=1):
    """ Find the first/best/last available checkpoints in directory dir.
    If n>1, it returns a list of first/best/last checkpoints, their epochs and devlosses."""

    all_checkpoints = np.sort(glob.glob("%s/model*.pth"%dir))
    mindevloss = 1000000000

    if len(all_checkpoints) > 0:

        if which == "last":
            checkpoints = all_checkpoints[-n:]
        elif which == "first":
            checkpoints = all_checkpoints[0:n]
        elif which == "best":
            all_devlosses = np.empty(len(all_checkpoints))
            for i, checkp in enumerate(all_checkpoints):
                if "_devloss_" in checkp:
                    all_devlosses[i] = float(re.sub(".pth", "", re.sub(".*_devloss_","",checkp)))
                else:
                    all_devlosses[i] = mindevloss
            sorti = np.argsort(all_devlosses)
            checkpoints = all_checkpoints[sorti[0:n]]

        epochs = [int(re.sub(".pth$", "", re.sub("_.*", "", re.sub(".*_epoch_","",ch)))) for ch in checkpoints]
        devlosses = [float(re.sub(".pth$", "", re.sub(".*_devloss_","",ch))) if "_devloss_" in ch else mindevloss for ch in checkpoints]
    else:
        checkpoints = [None]
        epochs = [0]
        devlosses = [mindevloss]

    if n==1:
        return checkpoints[0], epochs[0], devlosses[0]
    else:
        return checkpoints, epochs, devlosses

def onehot_for_class_ids(sample_class_idxs, target_ids, map_ids_to_idxs):
    """ One-hot vector but with the columns sorted based on the list of classes in
    target_ids. Since sample_classidxs contains indices, the map is used to go from
    indices to class ids."""
    onehot = np.zeros((sample_class_idxs.shape[0], len(target_ids)))

    if type(sample_class_idxs) == torch.Tensor:
        # I should eventually figure out if I can do this in a way that
        # works for both tensors and np arrays so that I don't need to move
        # back and forth from torch to numpy
        device = sample_class_idxs.device
        sample_class_idxs = sample_class_idxs.detach().cpu().numpy()
    else:
        device = None

    for i, tid in enumerate(target_ids):
        if tid in map_ids_to_idxs:
            # tid might not be in the map when enrolling new classes into a pre-existing model
            onehot[:,i] = np.array(sample_class_idxs==map_ids_to_idxs[tid], dtype=int)
        else:
            print ("Warning: class %s not present in the metadata maps"%tid)
        
    if device:
        onehot = np_to_torch(onehot, device)

    return onehot


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def np_to_torch(x, device):
    return torch.from_numpy(x).type(torch.get_default_dtype()).to(device)


def load_key_delete(keyfile, ids):

    # Create a matrix where the rows and columns correspond to the ids in the ids array
    # and the entries are given by the keyfile
    # 1: for target trials
    # 0: for impostor trials
    # -1: for unused trials

    enroll_ids, test_ids, labels = np.array([l.strip().split() for l in open(keyfile).readlines()]).T
    
    missing = set.difference(set(enroll_ids),set(ids))
    if len(missing)>0:
        print("There are %d enrollment ids in the key missing from embeddings file. Ignoring them."%len(missing))

    missing = set.difference(set(test_ids),set(ids))
    if len(missing)>0:
        print("There are %d test ids in the key missing from embeddings file. Ignoring them."%len(missing))

    str_to_idx = dict(zip(ids, np.arange(len(ids))))
    mask = -1*np.ones([len(ids), len(ids)])
    labels_int = np.array(labels=='tgt', dtype=int)

    for e, t, l in zip(enroll_ids, test_ids, labels_int):
        if e in str_to_idx and t in str_to_idx:
            mask[str_to_idx[e],str_to_idx[t]] = l
            mask[str_to_idx[t],str_to_idx[e]] = l

    return mask


def get_parameters_to_train(model, freeze_params=None):

    include_parameters = []
    for k, p in model.named_parameters():
        if freeze_params is None:
            include_parameters.append(k)
        else:
            freeze = False
            for fp in freeze_params:
                if re.search(fp, k) is not None:
                    freeze = True
            if not freeze:
                include_parameters.append(k)

    groups = []
    params = []

    print("Training parameters:")
    for n, p in model.named_parameters():
        if n in include_parameters:
            print("  %s"%n)
            params.append(p)

    if len(params) > 0:
        groups.append({'params': params})
    else:
        raise Exception("No parameters to train")


    return groups


def expand_regexp_dict(named_parameters, regexp_dict, name='l2_reg'):

    if regexp_dict is None:
        return None

    expanded_dict = dict()

    # First assign the cases that have specific values
    print("Expanded %s dictionary"%name)
    for k, v in regexp_dict.items():
        if k != "default":
            for n in named_parameters:
                if re.search(k, n) is not None:
                    print("  %s: %f"%(n, v))
                    expanded_dict[n] = v

    # For the rest, assign the default value
    v = regexp_dict['default']
    for n in named_parameters:
        if n not in expanded_dict:
            if 'default' not in regexp_dict:
                raise Exception("Default value missing from dictionary")
            print("  %s: %f"%(n, v))
            expanded_dict[n] = v

    return expanded_dict


def l2_reg_term(named_params, l2_reg_dict):

    term = 0
    for n, p in named_params:
        term += l2_reg_dict[n] * torch.pow(p, 2).sum() / 2.0

    return term


class CholInv:
    """
    Ci = CholInv(C) conveniently encapsulates the Cholesky transform of C, so that
    Ci @ RHS is equivalent to solve(C,RHS).
    """
    def __init__(self,C):
        self.L = chol(C)       # left Cholesky factor of C (lower triangular)
        self.C = C
        
    def __matmul__(self,RHS):
        """
        Does the equivalent of solve (C,RHS), using the stored Cholesky 
        transform. Can be conveniently invoked as self @ RHS.
        """
        L = self.L
        return solve_triangular(L, solve_triangular(L, RHS, trans=0, lower=True, check_finite=False),
                                lower=True, trans=1, check_finite=False)
    
    def logdet(self):
        """
        log determinant of inv(C) = - log determinant of C
        """
        return -2*np.log(np.diag(self.L)).sum()


def compute_weights_to_balance_by_class(class_names):

    unique_classes, class_idxs = np.unique(class_names, return_inverse=True)
    count = np.bincount(class_idxs)
    weight = 1.0/count
    weight *= len(weight)/np.sum(weight)
    return weight[class_idxs]


def print_weights_by_class(class_names, sample_weights):

    unique_classes, class_idxs = np.unique(class_names, return_inverse=True)
    for i, v in enumerate(unique_classes):
        ws = sample_weights[class_idxs==i] 
        w = ws.sum()
        c = len(ws)
        all_same = int(np.all(ws==ws[0]))
        print("Weight for %-40s = %5.2f (all same = %d), sum = %8.2f, number of samples = %8d"%(v, ws[0], all_same, w, c))


def ismember(a, b):
    """ A replica of the MATLAB function """
    b = dict((e, i) for i, e in enumerate(b))
    rindex = np.array([b.get(k, -1) for k in a])
    return rindex != -1, rindex



def get_class_to_cluster_map_from_config(config):

    if config.get("hierarchical"):
        if config.get("class_to_cluster_map"):
            class_to_cluster_map = config.get("class_to_cluster_map")
        else:
            class_to_cluster_map = dict([l.strip().split() for l in open(config.get("fixed_enrollment_ids")).readlines()])
    else:
        class_to_cluster_map = None

    return class_to_cluster_map


