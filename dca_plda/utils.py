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

def train(model, loader, optimizer, epoch, config, debug_dir=None):
    
    model.train()
    total_loss = total_regt = total_batches = 0
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
        loss = modules.compute_loss(output, metadata=metadata, ptar=config.ptar, loss_type=config.loss) 
        loss_scale = config.batch_size * 0.056 
        regt = l2_reg_term(model.named_parameters(), config.l2_reg_dict)
        regloss = loss_scale * loss + regt 
        regloss.backward()
        if config.get('max_norm') is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)
        optimizer.step()

        total_loss += loss
        total_regt += regt
        total_batches += 1
        if batch_idx % freq == 0 and batch_idx>0:
            print("  Epoch %04d, batch %04d, ave loss %f, ave reg term %f"%(epoch, batch_idx, total_loss/total_batches, total_regt/total_batches/loss_scale))
    
    print("Finished Epoch %04d, ave loss %f, ave reg term %f"%(epoch, total_loss/total_batches, total_regt/total_batches/loss_scale))

    if debug_dir:
        batches_file.close()

    return total_loss/total_batches


def evaluate(model, dataset, emap, tmap, min_dur=0):
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
        if np.any(ids!=tmap.sample_ids) or np.any(ids!=emap.sample_ids):
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
                scoreski = model.score(data[eidx], data[tidx], durs[eidx], durs[tidx])
                scoreski[durs[eidx]<min_dur, :] = 0
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


def save_checkpoint(model, outdir, epoch, trn_loss=None, dev_loss=None, optimizer=None, scheduler=None, name="model"):
    
    if trn_loss is not None:
        lr = optimizer.param_groups[0]['lr']
        outfile = '%s/%s_epoch_%04d_lr_%8.6f_trnloss_%.4f_devloss_%.4f.pth' % (outdir, name, epoch, lr, trn_loss, dev_loss)
    else:   
        if dev_loss is not None:
            outfile = '%s/%s_epoch_%04d_devloss_%.4f.pth' % (outdir, name, epoch, dev_loss)
        else:
            outfile = '%s/%s_epoch_%04d.pth' % (outdir, name, epoch)

    print("Saving model in %s"%outfile)
    save_dict = {'model': model.state_dict(), 
                 'config': model.config,
                 'optimizer': optimizer.state_dict() if optimizer is not None else None,
                 'scheduler': scheduler.state_dict() if scheduler is not None else None}
    torch.save(save_dict, outfile)

    return outfile


def load_model(file, device):

    loaded_dict = torch.load(file, map_location=device)
    config = loaded_dict['config']
    in_size = loaded_dict['model']['lda_stage.W'].shape[0]

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


def find_checkpoint(dir, which="last"):
    # Find the last available checkpoint in directory dir
    checkpoints = np.sort(glob.glob("%s/model*.pth"%dir))
    mindevloss = 1000000000

    if len(checkpoints) > 0:

        if which == "last":
            checkpoint = checkpoints[-1]
        elif which == "first":
            checkpoint = checkpoints[0]
        elif which == "best":
            besti = 0
            for i, checkp in enumerate(checkpoints):
                devloss = float(re.sub(".pth", "", re.sub(".*_devloss_","",checkp)))
                if devloss < mindevloss:
                    mindevloss = devloss
                    besti = i
            checkpoint = checkpoints[besti]

        epoch = re.sub("_.*", "", re.sub(".*_epoch_","",checkpoint))
        devloss = float(re.sub(".pth", "", re.sub(".*_devloss_","",checkpoint)))
    else:
        checkpoint = None
        epoch = 0
        devloss = mindevloss

    return checkpoint, int(epoch), devloss


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
