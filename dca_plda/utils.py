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
from dca_plda import scores
from itertools import groupby

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


def create_scoring_masks(metadata, class_name='class_id', num_models_dict=None):
    """"
    Create a scoring mask for batch consisting of the given metadata.
    If num_models_dict is provided, multi-side enrollment trials are provided as requested. The dict
    should consist of number of enrollment samples as keys and the number of models to be created 
    for each speaker for that case. Eg: num_models_dict = {2: 5, 3: 5} will create 5 2-side models
    and 5 3-side models for each speaker in the batch. All 1side-models are always created.
    """

    if num_models_dict is None:

        def _same_val_mat(f):
            ids = metadata[f].unsqueeze(1)
            return ids == ids.T
    
        same_cla = _same_val_mat(class_name)
        same_dom = _same_val_mat('domain_id')
        same_ses = _same_val_mat('session_id')

        emap = None

    else:

        def _same_val_mat(f1, f2):
            return f1.unsqueeze(1) == f2.unsqueeze(1).T

        # First, make sure that the class_ids are sorted as expected, with N consecutive samples from
        # each speaker. 
        class_ids = metadata[class_name]
        domain_ids = metadata['domain_id']
        session_ids = metadata['session_id']

        batch_size = class_ids.shape[0]

        # Diagonal emap corresponding to the 1side model above with corresponding same_* matrices
        list_emap     = [torch.eye(batch_size)]
        list_same_cla = [_same_val_mat(class_ids, class_ids)]
        list_same_dom = [_same_val_mat(domain_ids, domain_ids)]
        list_same_ses = [_same_val_mat(session_ids, session_ids)]
        key = (list_same_cla[0].clone().detach().int()*2-1) * list_same_dom[0] * ~list_same_ses[0]
        print("Number of trials for 1-side = %d / %d"%(torch.sum(key==1), torch.sum(key==-1)))

        # The number of consecutive samples per class has to be a multiple of the number of samples
        # per class set in the config. Yet, we do not have access to that info here so we take it to
        # be the minimum number of consecutive samples per class. In some instances, two sets of 
        # samples from the same class may be selected consecutively, so we check that all numbers
        # in the array are multiple of the minimum.
        consecutive_samples_per_class = np.unique([sum(1 for e in group) for _, group in groupby(class_ids)])
        if np.any(np.array(consecutive_samples_per_class)%np.min(consecutive_samples_per_class)!=0):
            raise Exception("Batch does not contain equal number of consecutive samples for each class")            
        else:
            consecutive_samples_per_class = np.min(consecutive_samples_per_class)

        num_classes = int(batch_size / consecutive_samples_per_class)

        for num_sides, num_models in num_models_dict.items():

            all_indices = []
            if num_sides >= consecutive_samples_per_class:
                raise Exception("Batch does not have enough samples per class to create %d-side models"%num_sides)

            for m in np.arange(num_models):
                # Create a list of indices indicating which samples to choose for each class
                relative_indices = np.sort(np.random.choice(np.arange(consecutive_samples_per_class), size = num_sides, replace=False))                # Repeat those indices as many times as classes in the batch
                indices = np.repeat(relative_indices[np.newaxis,:], num_classes, axis=0)
                # Now add the starting index of each class
                start_index = np.arange(0, batch_size, consecutive_samples_per_class)
                indices += start_index[:,np.newaxis]
                all_indices.append(indices)

            all_indices = np.concatenate(all_indices)
            emap_mside = torch.zeros([num_classes*num_models, batch_size])
            emap_mside[np.repeat(np.atleast_2d(np.arange(emap_mside.shape[0])), num_sides,axis=0).T, all_indices] = 1
            list_emap.append(emap_mside)

            # The class is the same for all samples in a model, so we can get it from the same class
            enroll_class_ids = class_ids[all_indices[:,0]]
            same_cla = _same_val_mat(enroll_class_ids, class_ids)
            list_same_cla.append(same_cla)

            # The domain may be different for different samples, but in most cases it will not, so
            # we just check against the domain of the first sample in the model
            enroll_domain_ids = domain_ids[all_indices[:,0]]
            same_dom = _same_val_mat(enroll_domain_ids, domain_ids)
            list_same_dom.append(same_dom)

            # Then, use the same_dom to sample some of the impostor samples so that they are not overrepresented
            # with respect to the target samples. This is because the number of target samples will be smaller
            # for the mside trials than for the 1side trials, but the number of impostor samples will be the same.
            # This is undesirable. Hence, we disable some percentage of the impostor samples to keep the ratio
            # of target to impostor samples in the mside trials the same as for the 1side trials.
            ratio_tgt = (consecutive_samples_per_class-num_sides)/(consecutive_samples_per_class-1)
            # Randomly turn to false that same fraction of trials
            mask = torch.ones_like(same_dom, dtype=torch.float).uniform_(0,1) < ratio_tgt
            # Keep all the target samples, only mask-out some of the impostor samples
            torch.logical_or(mask, same_cla, out=mask)
            torch.logical_and(same_dom, mask, out=same_dom)

            # Finally, we do need to check that all the sessions included in a model are different
            # from the test session. 
            same_ses = torch.zeros_like(same_cla, dtype=torch.bool)
            for i in np.arange(num_sides):
                enroll_sess_ids_i = session_ids[all_indices[:,i]]
                same_ses_i = _same_val_mat(enroll_sess_ids_i, session_ids)
                torch.logical_or(same_ses, same_ses_i, out=same_ses)
            list_same_ses.append(same_ses)

            # Temp code to check the number of trials of each kind for each num_sides
            valid = same_dom * ~same_ses
            key = (same_cla.clone().detach().int()*2-1)*valid
            print("Number of trials for %d-sides = %d / %d"%(num_sides, torch.sum(key==1), torch.sum(key==-1)))

        emap = torch.cat(list_emap).T
        same_cla = torch.cat(list_same_cla)
        same_dom = torch.cat(list_same_dom)
        same_ses = torch.cat(list_same_ses)

    # Valid trials are those with same domain and different session. The other trials are not scored.
    valid = same_dom * ~same_ses
    key = (same_cla.clone().detach().int()*2-1)*valid

    return key, emap


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


def print_batch(outf, metadata, maps, batch_num, key=None, emap=None):

    metadata_str = dict()
    for k in metadata.keys():
        if k in ['sample_id', 'class_id', 'session_id', 'domain_id']:
            v = metadata[k].detach().cpu().numpy()
            metadata_str[k] = np.atleast_2d([maps[k][i] for i in v])

    batch_str = np.ones_like(metadata_str['sample_id'])
    batch_str[:] = str(batch_num)
    np.savetxt(outf, np.concatenate([batch_str] + list(metadata_str.values())).T, fmt="%s")

    if key is not None:
        ids = metadata_str['sample_id'][0]
        if emap is not None:
            enroll_ids = ["%s_model%s"%(f[0], f[1]) for f in np.c_[metadata_str['class_id'][0][np.array([np.where(line==1)[0][0] for line in emap.T])], np.arange(0,emap.shape[1])]]
            scores.Key(enroll_ids, ids, key.detach().cpu().numpy()).save("%s.%d"%(outf.name,batch_num), 'ascii')
            emapf = open("%s.%d.emap"%(outf.name,batch_num),'w')
            for i, m in enumerate(emap.T):
                for j in np.where(m==1)[0]:
                    emapf.write("%s %s\n"%(enroll_ids[i], ids[j]))
            emapf.close()
        else:
            ids = metadata_str['sample_id'][0]
            scores.Key(ids, ids, key.detach().cpu().numpy()).save("%s.%d"%(outf.name,batch_num), 'ascii')
                            
                                                                                        
