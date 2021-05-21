import torch
from IPython import embed
import numpy as np
import sklearn.metrics
import configparser
import glob
import re
import modules 
import os
import scores as scr
from numpy.linalg import cholesky as chol
from scipy.linalg import solve_triangular

def train(model, loader, optimizer, epoch, config, debug_dir=None):
    
    model.train()
    total_loss = total_regt = total_regl = total_batches = 0
    freq = 100

    if debug_dir:
        batches_file = open("%s/batches_epoch_%04d"%(debug_dir, epoch), "w")

    print("Starting epoch %d"%epoch)

    for batch_idx, (data, metadata) in enumerate(loader):
        if debug_dir:
            print_batch(batches_file, metadata, loader.metamaps, batch_idx)
        durs = metadata['duration']
        output = model(data, durs)
        optimizer.zero_grad()

        # Make the loss proportional to the batch size so that the regularization
        # parameters can be kept the same if batch size is changed. This assumes
        # that bigger batches mean more robust loss (more or less linearly since
        # that is how the number of target trials grows with batch size). Note that
        # this is just an approximation. In practice, the optimal regularization
        # parameters might still have to be retuned after changing the batch size.
        # Note I am not using the weight_decay parameter in the optimizer because
        # I want to do the clipping on the full gradient rather than only on the 
        # part corresponding to the loss since this seems to work better.
        # The loss_scale is there for backward compatibility with my old TF code.
        loss, llrs, labels = modules.compute_loss(output, metadata=metadata, ptar=config.ptar, loss_type=config.loss, 
            return_info=True, enrollment_ids=model.enrollment_classes, ids_to_idxs=loader.metamaps['speaker_id_inv'])

        loss_scale = config.batch_size * 0.056 
        regt, regl = 0.0, 0.0

        l2_reg_dict = config.get("l2_reg_dict")
        if l2_reg_dict is not None:
            regt = l2_reg_term(model.named_parameters(), config.l2_reg_dict)
        
        llr_reg_dict = config.get("llr_reg")
        if llr_reg_dict is not None:
            # Regularize the llrs directly as proposed in "Gradient Starvation:A Learning Proclivity in Neural Networks" 
            # by Pezeshki et al
            regl = llr_reg_dict['weight'] * torch.pow(llrs-llr_reg_dict.get('offset',0.0), 2).sum() 
        
        regloss = loss_scale * loss + regt + regl
        regloss.backward()
        if config.get('max_norm') is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)
        optimizer.step()

        total_loss += loss.detach()
        if l2_reg_dict is not None:
            total_regt += regt.detach() 
        if llr_reg_dict is not None:
            total_regl += regl.detach() 

        total_batches += 1
        if batch_idx % freq == 0 and batch_idx>0:
            print("  Epoch %04d, batch %04d, ave loss %f, ave l2 reg term %f, ave logit reg term %f"%
                (epoch, batch_idx, total_loss/total_batches, 
                total_regt/total_batches/loss_scale,
                total_regl/total_batches/loss_scale))
    
    print("Finished Epoch %04d, ave loss %f, ave l2 reg term %f, ave logit reg term %f"%
        (epoch, total_loss/total_batches, total_regt/total_batches/loss_scale, total_regl/total_batches/loss_scale))

    if debug_dir:
        batches_file.close()

    return total_loss/total_batches


def evaluate(model, dataset, emap, tmap, min_dur=0, raw=False):
    """ Evaluate the model on the embeddings corresponding to emap and tmap.
    The code assumes that tmap only has single-session test models.
    For the multi-session enrollment models, we simply create single-session trials
    and then average the resulting scores. 
    """
    if len(tmap.mappings.keys()) > 1 or list(tmap.mappings.keys())[0]!=1:
        raise Exception("Scoring not implemented for multi-session test maps.")

    data = dataset.get_data()
    durs = dataset.get_durs()
    ids  = dataset.get_ids()
    model.eval()

    with torch.no_grad():
        
        # We assume that emap and tmap were created using the ids in the dataset, but
        # we double check just in case.
        if np.any(ids!=tmap.sample_ids) or (np.any(ids!=emap.sample_ids) and model.enrollment is None):
            raise Exception("The dataset is not sorted the same way as the columns in emap or tmap. Make sure emap and tmap are created using the sample_ids in the dataset object you use in evaluation.")

        tidx = tmap.mappings[1]['map'][0]
        tids = tmap.model_ids
        scores = []
        eids   = []
        for k in emap.mappings.keys():
            emapk = emap.mappings[k]['map']
            eidsk = emap.mappings[k]['model_ids']
            eids += eidsk
            scoresk = torch.zeros(len(eidsk), len(tids))  
            for i in np.arange(k):
                eidx = emapk[i]
                if model.enrollment is None:
                    scoreski = model.score(data[tidx], data[eidx], durs[tidx], durs[eidx], raw=raw)
                    scoreski[durs[eidx]<min_dur, :] = 0
                else:
                    # Enrollment embeddings are part of the model
                    scoreski = model.score(data[tidx], None, durs[tidx], None, raw=raw)[eidx]                    

                scoreski[:, durs[tidx]<min_dur] = 0
                scoresk += scoreski
            scoresk /= k
            scores.append(scoresk)

        scores = torch.cat(scores,0)
        # Make sure that the enrollment ids are sorted as in the emap, else, other stuff might break.
        # If there are no bugs this check should always pass.
        if np.any(eids != emap.model_ids):
            raise Exception("Something is wrong, model_ids in emap do not coincide with the model ids after scoring")

    return scr.Scores(eids, tids, scores)


def compute_sideinfo(model, dataset):

    data = dataset.get_data()
    ids  = dataset.get_ids()
    model.eval()

    with torch.no_grad():
        si = model.score(data, si_only=True)

    return ids, si
   


def print_batch(outf, metadata, maps, batch_num):

    metadata_str = dict()
    for k in metadata.keys():
        if k in ['sample_id', 'speaker_id', 'session_id', 'domain_id']:
            v = metadata[k].detach().cpu().numpy()
            metadata_str[k] = np.atleast_2d([maps[k][i] for i in v])

    batch_str = np.ones_like(metadata_str['sample_id'])
    batch_str[:] = str(batch_num)
    np.savetxt(outf, np.concatenate([batch_str] + list(metadata_str.values())).T, fmt="%s")


def print_graph(model, dataset, device, outdir):

    # Print the graph using torchviz. 
    try:
        from torchviz import make_dot
        outfile = "%s/graph"%outdir
        print('Printing graph in %s'%outfile)
        loader = ddata.TrialLoader(dataset, device, seed=0, batch_size=10, num_batches=1)
        x, meta_batch = next(loader.__iter__())
        x.requires_grad_(True)
        durs = meta_batch['duration'].requires_grad_(True)
        output = model(x, durs)
        params = dict(list(model.named_parameters()) + [('x', x),('durs', durs)])
        make_dot(output, params=params).render(outfile, format="png")
    except:
        print("Torchviz unavailable. Skipping graph creation")

    print("List of model parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("  ", name, param.shape)


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


def load_model(file, device):

    loaded_dict = torch.load(file, map_location=device)
    config = loaded_dict['config']

    if 'in_dim' in loaded_dict:
        in_size = loaded_dict['in_dim']
    else:
        # Backward compatibility with old models that did not save in_dim
        if 'front_stage.W' in loaded_dict['model']:
            # If there is a front stage, get the in_size from there
            in_size = loaded_dict['model']['front_stage.W'].shape[0]
        elif 'lda_stage.W' in loaded_dict['model']:
            # Get it from the lda stage
            in_size = loaded_dict['model']['lda_stage.W'].shape[0]
        elif 'plda_stage.F' in loaded_dict['model']:
            in_size = loaded_dict['model']['plda_stage.F'].shape[0]
        else:
            raise Exception("Cannot infer input dimension for this model")

    model = modules.DCA_PLDA_Backend(in_size, config)
    model.load_state_dict(loaded_dict['model'])

    return model


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


def create_scoring_masks(metadata):

    def _same_val_mat(f):
        ids = metadata[f].unsqueeze(1)
        return ids == ids.T
    
    same_spk = _same_val_mat('speaker_id')
    same_dom = _same_val_mat('domain_id')
    same_ses = _same_val_mat('session_id')
    # Valid trials are those with same domain and different session. The other trials are not scored.
    # While the batches are created so that all target trials come from different sessions, some of the
    # impostor trials could be same-session in some datasets (ie, when a session involves two sides of
    # a phone conversation). We need to discard those trials here.
    valid = same_dom * ~same_ses

    return same_spk, valid


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

def mkdirp(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)

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
                all_devlosses[i] = float(re.sub(".pth", "", re.sub(".*_devloss_","",checkp)))
            sorti = np.argsort(all_devlosses)
            checkpoints = all_checkpoints[sorti[0:n]]

        epochs = [int(re.sub("_.*", "", re.sub(".*_epoch_","",ch))) for ch in checkpoints]
        devlosses = [float(re.sub(".pth", "", re.sub(".*_devloss_","",ch))) for ch in checkpoints]
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
        for i, tid in enumerate(target_ids):
            onehot[:,i] = np.array(sample_class_idxs.detach().cpu().numpy()==map_ids_to_idxs[tid], dtype=int)
        onehot = np_to_torch(onehot, sample_class_idxs.device)
    else:
        for i, tid in enumerate(target_ids):
            onehot[:,i] = np.array(sample_class_idxs==map_ids_to_idxs[tid], dtype=int)

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


def ismember(a, b):
    """ A replica of the MATLAB function """
    b = dict((e, i) for i, e in enumerate(b))
    rindex = np.array([b.get(k, -1) for k in a])
    return rindex != -1, rindex


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

