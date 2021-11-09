# This file includes methods that are called from the wrapper scripts or that
# require imports from the repo itself or that are only needed by one of those
# methods.

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

from dca_plda import modules 
from dca_plda import data as ddata
from dca_plda.scores import IdMap, Key, Scores
from dca_plda import calibration 
from dca_plda.utils import *


def mkdirp(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)


def setup_torch_and_cuda_vars(use_cuda):
    cudnn.benchmark = True
    if use_cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        device = torch.device("cuda:0")
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
        device = torch.device("cpu")
    return device


def test_model(model, data_dict, model_name, ptar, loss_type, print_min_loss=False, level1_loss_weight=None):
    # Compute the loss for each of the devsets. The validation loss will be the average over all devsets
    av_loss = 0
    for name, info in data_dict.items():

        mask = info['mask']
        scores = evaluate(model, info['dataset'], info['emap'], info['tmap'])
        # Since the key mask was created to be aligned to the emap and tmap, the score_mat
        # is already aligned to this mask
        loss, llrs, labels = modules.compute_loss(scores.score_mat, mask=mask, ptar=ptar, loss_type=loss_type, return_info=True)

        if level1_loss_weight:
            # Compute the loss for the inner level of the hierarchical model and add it to the loss
            mask_level1 = info['mask_level1']
            scores_level1 = evaluate(model, info['dataset'], info['emap_level1'], info['tmap'], level='level1')
            loss_level1, _, _ = modules.compute_loss(scores_level1.score_mat, mask=mask_level1, ptar=ptar, loss_type=loss_type, return_info=True)
            loss_level2 = loss
            if level1_loss_weight < 1.0:
                loss = level1_loss_weight * loss_level1 + (1-level1_loss_weight) * loss_level2
            else:
                loss = loss_level1
        if print_min_loss:
            tar = llrs[labels==1].detach().cpu().numpy()
            non = llrs[labels==0].detach().cpu().numpy()
            cal = calibration.logregCal(tar, non, ptar)
            min_loss = cross_entropy(cal(tar), cal(non), ptar)
            print("%s, loss on dev set %s = %f  min = %f"%(model_name, name, loss, min_loss))
        else:
            if level1_loss_weight:
                print("%s, loss on dev set %-20s = %.4f (level1 = %.4f, level2 = %.4f)"%(model_name, name, loss, loss_level1, loss_level2))   
            else:
                print("%s, loss on dev set %-20s = %.4f"%(model_name, name, loss))
        av_loss += loss.detach()

    av_loss /= len(data_dict)
    return av_loss

def load_data_dict(table, device, fixed_enrollment_ids=None, map_enrollment_ids_to_level1=None, fixed_enrollment_ids_level1=None):
    data_dict = dict()
    for line in open(table).readlines():
        f = line.strip().split()
        name, emb, key, emapf, tmapf = f[0:5]
        dur = f[5] if len(f) > 3 else None
        dataset = ddata.LabelledDataset(emb, dur, meta_is_dur_only=True, device=device)
        if fixed_enrollment_ids is not None:
            # Enrollment embeddings are part of the model, not provided in the data
            # Hence, emapf in this case should either be NONE, in which case, all
            # detectors are used, or some subset of fixed_enrollment_ids, in which
            # case only that subset is used
            emap = IdMap.load(emapf, fixed_enrollment_ids)
        else:
            emap = IdMap.load(emapf, dataset.get_ids())

        tmap = IdMap.load(tmapf, dataset.get_ids())
        # Load key file in the order in which the model ids were loaded in emap 
        # and tmap. This ensures that the scores and keys will be aligned.
        mask = np_to_torch(Key.load(key, emap.model_ids, tmap.model_ids).mask, device)

        if map_enrollment_ids_to_level1 is not None:
            # Derive the mask for level1 from the map between level1 and output fixed_enrollment_ids
            # and the output mask.
            mask_level1 = -1*torch.ones([len(fixed_enrollment_ids_level1), mask.shape[1]])
            for i, cluster in enumerate(fixed_enrollment_ids_level1):
                idxs = np.array([emap.model_ids.index(l) for l, c in map_enrollment_ids_to_level1.items() if c==cluster])
                mask_level1[i,torch.where(mask[idxs]==1)[1]] = 1
            emap_level1 = IdMap.load('NONE', fixed_enrollment_ids_level1)    
            #mask_level1 = np_to_torch(mask_level1, device)
        else:
            mask_level1 = None
            emap_level1 = None
        
        data_dict[name] = {'dataset': dataset, 'mask': mask, 'emap': emap, 'tmap': tmap, 
        'mask_level1': mask_level1, 'emap_level1': emap_level1}

    return data_dict


def train(model, trn_dataset, config, dev_table, out_dir, device, 
    seed=0, restart=False, debug=False, init_subset=None, print_min_loss=False):

    if hasattr(model, 'class_to_cluster_map') and config.get('level1_loss_weight') is not None:
        dev_data_dict = load_data_dict(dev_table, device, model.enrollment_classes, 
            map_enrollment_ids_to_level1=model.class_to_cluster_map,
            fixed_enrollment_ids_level1=model.level1_detector.enrollment_classes)
    else:
        dev_data_dict = load_data_dict(dev_table, device, model.enrollment_classes)
        
    param_names = [k for k, v in model.named_parameters()]
    config['l2_reg_dict'] = expand_regexp_dict(param_names, config.l2_reg)
    num_epochs = config.num_epochs

    # Initialize the optimizer and learning rate scheduler
    parameters = get_parameters_to_train(model, config.get('freeze_params'))
    optimizer = optim.Adam(parameters, lr=config.learning_rate, betas=config.betas)
    lr_scheduler = None

    # For backward compatibility with older configs where the names were different, and the value
    # True corresponded to same_num_classes_per_dom_then_same_num_samples_per_class
    bbd  = config.get("balance_method_for_batches", config.get("balance_batches_by_domain", 'none'))
    bbdi = config.init_params.get("balance_method", config.init_params.get("balance_by_domain", 'none'))
    config["balance_method_for_batches"] = bbd  if bbd  is not True else 'same_num_classes_per_dom_then_same_num_samples_per_class'
    config.init_params["balance_method"] = bbdi if bbdi is not True else 'same_num_classes_per_dom_then_same_num_samples_per_class'

    if config.get("learning_rate_params") is not None:
        conf = config.learning_rate_params
        lrtype = conf['type']
        factor = conf['factor']

        if lrtype == 'CyclicLR':
            lr = config.learning_rate
            lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, lr/factor, lr, cycle_momentum=False, step_size_up=conf['step_size_up'], step_size_down=conf['step_size_down'])
        else:
            patience = conf['patience']
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience, threshold=0.001, min_lr=0.000001) 


    # Load the last available model if restart is set or initialize with data if no previous checkpoint is available
    last_checkpoint, last_epoch, last_dev_loss = find_checkpoint(out_dir)

    if restart and last_checkpoint is not None:

        print("Loading model from %s"%last_checkpoint)

        if last_epoch == 0:
            # We are loading the zeroth epoch which might be from a prior stage or just the initialization with data, 
            # so, do not load the optimizer's or lr_scheduler check point in this case
            load_checkpoint(last_checkpoint, model, device)
        else:           
            load_checkpoint(last_checkpoint, model, device, optimizer, lr_scheduler)

        best_checkpoint,  best_epoch,  best_dev_loss  = find_checkpoint(out_dir, "best")
        first_checkpoint, first_epoch, first_dev_loss = find_checkpoint(out_dir, "first")

    else:

        if init_subset:
            print("Initializing model using a subset of the training data: %s"%init_subset)
            init_subset = [l.strip().split()[0] for l in open(init_subset).readlines()]
        else:
            print("Initializing model using all training data")
            init_subset = None

        embeddings, metadata, metamaps = trn_dataset.get_data_and_meta(init_subset)
        trn_loss = model.init_params_with_data(embeddings, metadata, metamaps, config, device=device)
        dev_loss = test_model(model, dev_data_dict, "Epoch 0000", config.ptar, config.loss, print_min_loss=print_min_loss, level1_loss_weight=config.get("level1_loss_weight"))
        checkpoint = save_checkpoint(model, out_dir, 0, trn_loss, dev_loss, optimizer, lr_scheduler)

        best_checkpoint,  best_epoch,  best_dev_loss  = checkpoint, 0, dev_loss
        first_checkpoint, first_epoch, first_dev_loss = checkpoint, 0, dev_loss


    # Create the trial loader
    embeddings, metadata, metamaps = trn_dataset.get_data_and_meta()
    loader = ddata.TrialLoader(embeddings, metadata, metamaps, device, seed=seed, batch_size=config.batch_size, num_batches=config.num_batches_per_epoch, 
                         balance_method=config.balance_method_for_batches, num_samples_per_class=config.num_samples_per_class)

    # Train the model
    start_epoch = 1 if not restart else last_epoch+1
    for epoch in range(start_epoch, num_epochs + 1):
        trn_loss = train_epoch(model, loader, optimizer, epoch, config, debug_dir=out_dir if debug else None)
        dev_loss = test_model(model, dev_data_dict, "Epoch %04d"%epoch, config.ptar, config.loss, print_min_loss=print_min_loss, level1_loss_weight=config.get("level1_loss_weight"))
        checkpoint = save_checkpoint(model, out_dir, epoch, trn_loss, dev_loss, optimizer, lr_scheduler)

        if dev_loss < best_dev_loss:
            best_checkpoint = checkpoint
            best_dev_loss = dev_loss
            best_epoch = epoch

        if lr_scheduler is not None:
            if lrtype == 'CyclicLR':
                lr_scheduler.step()
            else:
                lr_scheduler.step(dev_loss)

    # Copy the first, best and the last models to files with explicit names 
    shutil.copy(checkpoint,       '%s/last_epoch_%04d_devloss_%.4f.pth' % (out_dir, epoch, dev_loss))
    shutil.copy(best_checkpoint,  '%s/best_epoch_%04d_devloss_%.4f.pth' % (out_dir, best_epoch, best_dev_loss))
    shutil.copy(first_checkpoint, '%s/first_epoch_%04d_devloss_%.4f.pth' % (out_dir, first_epoch, first_dev_loss))

    if config.compute_ave_model:
        # Average the best/first N models and retest. Stop when the performance starts degrading
        checkpoints, _, devlosses = find_checkpoint(out_dir, config.compute_ave_model, n=num_epochs)
        cur_ave_dev_loss = devlosses[0]
        prev_ave_dev_loss = 100000
        load_checkpoint(checkpoints[0], model, device)
        print("Starting averaging of models with %s"%checkpoints[0])
        n = 1
        while cur_ave_dev_loss < prev_ave_dev_loss:
            print("Updating average with %s"%checkpoints[n])
            update_model_with_weighted_average(model, checkpoints[n], device, n)
            prev_ave_dev_loss = cur_ave_dev_loss
            cur_ave_dev_loss = test_model(model, dev_data_dict, "Average %s n=%03d"%(config.compute_ave_model,n), config.ptar, config.loss, print_min_loss=print_min_loss)
            print("New devloss = %f"%cur_ave_dev_loss)
            n += 1
        save_checkpoint(model, out_dir, dev_loss=dev_loss, name="ave_%s"%config.compute_ave_model)
        
    Path("%s/DONE"%out_dir).touch()




def train_epoch(model, loader, optimizer, epoch, config, debug_dir=None):
    
    model.train()
    total_loss = total_regt = total_regl = total_batches = total_loss_level1 = 0
    freq = 100

    if debug_dir:
        batches_file = open("%s/batches_epoch_%04d"%(debug_dir, epoch), "w")

    print("Starting epoch %d"%epoch)
    level1_loss_weight = config.get("level1_loss_weight")
    if level1_loss_weight is not None:
        print("Using loss weight for level 1 = %f"%level1_loss_weight)

    for batch_idx, (data, metadata) in enumerate(loader):

        if debug_dir:
            print_batch(batches_file, metadata, loader.metamaps, batch_idx)

        durs = metadata['duration']
        if level1_loss_weight:
            output, output_level1, _ = model(data, durs, all_levels=True)
        else:
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
        loss, llrs, _ = modules.compute_loss(output, metadata=metadata, ptar=config.ptar, loss_type=config.loss, 
            return_info=True, enrollment_ids=model.enrollment_classes, ids_to_idxs=loader.metamaps['class_id_inv'])

        if level1_loss_weight:
            # Compute the loss for the inner level of the hierarchical model and add it to the loss
            loss_level1, _, _ = modules.compute_loss(output_level1, metadata=metadata, ptar=config.ptar, loss_type=config.loss, 
            return_info=True, enrollment_ids=model.level1_detector.enrollment_classes, ids_to_idxs=loader.metamaps['cluster_id_inv'], class_name='cluster_id')
            if level1_loss_weight < 1.0:
                loss = level1_loss_weight * loss_level1 + (1-level1_loss_weight) * loss
            else:
                loss = loss_level1
        else:
            loss_level1 = 0.0

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
        if level1_loss_weight:
            total_loss_level1 += loss_level1.detach()
        if l2_reg_dict is not None:
            total_regt += regt.detach() 
        if llr_reg_dict is not None:
            total_regl += regl.detach() 

        total_batches += 1
        if batch_idx % freq == 0 and batch_idx>0:
            print("  Epoch %04d, batch %04d, ave loss %f, ave l2 reg term %f, ave loss level1 %f"%
                (epoch, batch_idx, total_loss/total_batches, 
                total_regt/total_batches/loss_scale,
                total_loss_level1/total_batches))
    
    print("Finished Epoch %04d, ave loss %f, ave l2 reg term %f, ave loss level1 %f"%
        (epoch, total_loss/total_batches, total_regt/total_batches/loss_scale, total_loss_level1/total_batches))

    if debug_dir:
        batches_file.close()

    return total_loss/total_batches


def evaluate(model, dataset, emap, tmap, min_dur=0, raw=False, level=None, cluster_prior_dict=None):
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

    if level is not None and not hasattr(model, 'level1_detector'):
        raise Exception("Level 1 or 2 scores can only be computed for hierarchical DCA-PLDA")

    with torch.no_grad():
        
        # We assume that emap and tmap were created using the ids in the dataset, but
        # we double check just in case.
        if np.any(ids!=tmap.sample_ids) or (np.any(ids!=emap.sample_ids) and model.enrollment_classes is None):
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
                if model.enrollment_classes is None:
                    scoreski = model.score(data[tidx], data[eidx], durs[tidx], durs[eidx], raw=raw)
                    scoreski[durs[eidx]<min_dur, :] = 0
                else:
                    # Enrollment embeddings are part of the model
                    if level is not None or cluster_prior_dict is not None:
                        scoreski_output, scoreski_level1, scoreski_level2 = model.score(data[tidx], None, durs[tidx], None, raw=raw, all_levels=True, cluster_prior_dict=cluster_prior_dict)
                    else:
                        scoreski_output = model.score(data[tidx], None, durs[tidx], None, raw=raw)

                    if level == 'level1':
                        scoreski = scoreski_level1[eidx]
                    elif level == 'level2':
                        scoreski = scoreski_level2[eidx]
                    else:
                        scoreski = scoreski_output[eidx]                    

                scoreski[:, durs[tidx]<min_dur] = 0
                scoresk += scoreski
            scoresk /= k
            scores.append(scoresk)

        scores = torch.cat(scores,0)
        # Make sure that the enrollment ids are sorted as in the emap, else, other stuff might break.
        # If there are no bugs this check should always pass.
        if np.any(eids != emap.model_ids):
            raise Exception("Something is wrong, model_ids in emap do not coincide with the model ids after scoring")

    return Scores(eids, tids, scores)


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
        if k in ['sample_id', 'class_id', 'session_id', 'domain_id']:
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
        emb, meta, idx_to_str = dataset.get_data_and_meta()
        loader = ddata.TrialLoader(emb, meta, idx_to_str, device, seed=0, batch_size=10, num_batches=1)
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

    if config.get("hierarchical"):
        model = modules.Hierarchical_DCA_PLDA_Backend(in_size, config)
    else:
        model = modules.DCA_PLDA_Backend(in_size, config)

    model.load_state_dict(loaded_dict['model'])

    return model

