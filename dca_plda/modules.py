import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
from IPython import embed
from sklearn import discriminant_analysis
from scipy.special import logit
from scipy.sparse import coo_matrix, dia_matrix
import utils 
import data as ddata 
import calibration
from scipy import linalg

def compute_loss(llrs, metadata=None, ptar=0.01, mask=None, loss_type='cross_entropy', return_info=False, 
    enrollment_ids=None, ids_to_idxs=None):

    if metadata is None:
        assert mask is not None
        # The mask is assumed to have been generated using the Key class, 
        # so targets are labeled with 1, impostors with -1 and non-scored trials with 0.
        valid = mask != 0
        same_spk = mask == 1
    else:
        assert metadata is not None
        if enrollment_ids is None:
            # The llrs are assumed to correspond to all vs all trials
            same_spk, valid = utils.create_scoring_masks(metadata)
        else:
            # The llrs are assumed to correspond to all samples. The speaker ids in this case are
            # indices that have to be mapped to enrollment indexes. 
            spk_ids = metadata['speaker_id'].type(torch.int)
            same_spk = utils.onehot_for_class_ids(spk_ids, enrollment_ids, ids_to_idxs).T
            # All samples are valid. 
            valid = torch.ones_like(llrs).type(torch.bool)

    # Select the valid llrs and shift them to convert them to logits
    llrs   = llrs[valid]
    logits = llrs + logit(ptar)
    labels = same_spk[valid]
    labels = labels.type(logits.type())

    ptart = torch.as_tensor(ptar)
    tarw = (labels == 1).float()
    impw = (labels == 0).float()
    
    if "weighted" in loss_type:
        # Weights are given by the inverse of the number of times that each enrollment model 
        # occurs in the data
        weights = 1/torch.sum(same_spk, axis=1, keepdim=True).float()
        weights *= len(weights)/sum(weights)
        weights = (weights * torch.ones_like(same_spk))[valid]
        tarw *= weights
        impw *= weights
    
    # The loss will be given by tar_weight * tar_loss + imp_weight * imp_loss
    tar_weight = tarw *    ptart /torch.sum(tarw) if torch.sum(tarw)>0 else 0.0
    imp_weight = impw * (1-ptart)/torch.sum(impw) if torch.sum(impw)>0 else 0.0

    # Finally, compute the loss and multiply it by the weight that corresponds to the impostors
    # Loss types are taken from Niko Brummer's paper: "Likelihood-ratio calibration using prior-weighted proper scoring rules"
    if loss_type == "cross_entropy" or loss_type == "weighted_cross_entropy":
        baseline_loss = -ptar*np.log(ptar) - (1-ptar)*np.log(1-ptar)
#        criterion = nn.BCEWithLogitsLoss(pos_weight=tar_weight/imp_weight, reduction='sum')
#        loss = criterion(logits, labels)*imp_weight/baseline_loss
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        losses = criterion(logits, labels)
        loss = torch.sum(tar_weight*losses + imp_weight*losses)/baseline_loss

    elif loss_type == "brier" or loss_type == "weighted_brier":
        baseline_loss = ptar * (1-ptar)**2 + (1-ptar) * ptar**2
        posteriors = torch.sigmoid(logits)
        loss = torch.sum(tar_weight*(1-posteriors)**2 + imp_weight*posteriors**2)/baseline_loss

    else:
        raise Exception("Unknown loss_type %s"%loss_type)

    if return_info:
        return loss, llrs, labels
    else:
        return loss


class DCA_PLDA_Backend(nn.Module):
    def __init__(self, in_dim, config):
        """
        Implements the PLDA branch which includes: a linear transform, length-norm and a PLDA-like transform
        """
        super().__init__()
        lda_dim = config.lda_dim
        self.config = config
        front_dim = self.config.get('front_dim')

        if front_dim:
            # Affine layer at the front (used to fine tune the 6th layer in the embedding extractor)
            self.front_stage = Affine(in_dim, front_dim)
            in_dim = front_dim

        self.lda_stage  = Affine(in_dim, lda_dim, "l2norm")

        self.plda_stage = Quadratic(lda_dim)

        if self.config.get('fixed_enrollment_ids'):
            # In this case, the enrollment vectors are pre-defined and another parameter of the model
            # Use the Affine class for this. Only the W of and the init methods of that class are used, not the forward.
            # Read the list of enrollment classes from a text file provided in the config.
            self.enrollment_classes = [l.strip().split()[0] for l in open(self.config.get('fixed_enrollment_ids')).readlines()]
            self.enrollment = Affine(lda_dim, len(self.enrollment_classes), bias=False)
        else:
            self.enrollment_classes = None
            self.enrollment = None

        if config.get('sideinfo_parameters'):
            idim = config.sideinfo_parameters.get('internal_dim', 200)
            edim = config.sideinfo_parameters.get('external_dim', 5)

            self.si_input = config.sideinfo_parameters.get('input_from','main_input') 
            if self.si_input == 'main_input':
                si_in_dim = in_dim
            elif self.si_input == 'plda_input':
                si_in_dim = lda_dim
            else:
                raise Exception("Unrecognized input for sideinfo branch")

            self.si_stage1 = Affine(si_in_dim, idim, "l2norm")

            stage2_activation = config.sideinfo_parameters.get('stage2_activation', 'logsoftmax')
            if config.sideinfo_parameters.get('stage2_inner_sizes'):
                # Pass the output of the first stage through a simple NN
                self.si_stage2 = SimpleNN(idim, edim, stage2_activation, config.sideinfo_parameters.get('stage2_inner_sizes'), config.sideinfo_parameters.get('stage2_inner_activation'))
            else:
                self.si_stage2 = Affine(idim, edim, stage2_activation)
        else:
            config.sideinfo_parameters = None
            edim = 0
        self.si_dim = edim

        if config.get('si_dependent_shift_parameters'):
            assert self.si_dim > 0
            # The shift selector is just a W matrix
            self.shift_selector = Affine(self.si_dim, lda_dim, activation=None, bias=False) 

        self.cal_stage  = CA_Calibrator(config.get('calibration_parameters'), si_dim=self.si_dim)

    def forward(self, x, durations=None):
        # The forward method assumes the same enroll and test data. This is used in training.
        return self.score(x, durt=durations)

    def score(self, xt, xe=None, durt=None, dure=None, si_only=False, raw=False):
        # Same as the forward method but it allows for assymetric scoring where the rows and columns
        # of the resulting score file corresponds to two different sets of data

        hase = xe is not None

        if hasattr(self, 'front_stage'):
            xe = self.front_stage(xe) if hase else None
            xt = self.front_stage(xt)

        x2e = self.lda_stage(xe) if hase else None
        x2t = self.lda_stage(xt) 
        
        if hasattr(self,'si_stage1'):
            si_inpute = xe if self.si_input == 'main_input' else x2e
            si_inputt = xt if self.si_input == 'main_input' else x2t
             
            s2e = self.si_stage1(si_inpute) if hase else None
            s2t = self.si_stage1(si_inputt) 

            sie = self.si_stage2(s2e) if hase else None
            sit = self.si_stage2(s2t) 

        else:
            sie = sit = None

        if si_only:
            assert xe is None and sit is not None
            return sit
        else:

            if self.config.get('si_dependent_shift_parameters'):
                # Use the si to find a shift vector for each sample 
                x2e = x2e - self.shift_selector(sie) if hase else None
                x2t = x2t - self.shift_selector(sit) 

            if self.enrollment is not None:
                assert xe is None and dure is None and sie is None
                scrs = self.plda_stage.score(x2t, self.enrollment.W.T)
                # In this case, there is no dur or si for the enrollment side. 
                # Set them to a vector of constants 
                dure = 2*torch.ones(scrs.shape[0]).to(scrs.device) if durt is not None else None
                sie = torch.ones([scrs.shape[0], sit.shape[1]]).to(scrs.device) if sit is not None else None
            else:
                scrs = self.plda_stage.score(x2t, x2e)

            llrs = self.cal_stage.calibrate(scrs, durt, sit, dure, sie)            
            return llrs if raw is False else scrs


    def init_params_with_data(self, dataset, config, device=None, subset=None):

        balance_by_domain = config.balance_batches_by_domain
        assert 'init_params' in config
        init_params = config.init_params
        
        # The code here repeats the steps in forward above, but adds the steps necessary for initialization.
        # I chose to keep these two methods separate to leave the forward small and easy to read.
        with torch.no_grad():

            x, meta, idx_to_str = dataset.get_data_and_meta(subset)
            speaker_ids = meta['speaker_id']
            domain_ids  = meta['domain_id']
            x_torch = utils.np_to_torch(x, device)
        
            if hasattr(self,'front_stage'):
                params_to_init_front_stage = config.get("params_to_init_front_stage")
                if params_to_init_front_stage:
                    # Read the params from an npz file
                    self.front_stage.init_with_params(np.load(params_to_init_front_stage))
                else:
                    self.front_stage.init_random(init_params.get('stdev',0.1))
                x_torch = self.front_stage(x_torch)
                x = x_torch.cpu().numpy()
                
            if init_params.get("random"):
                self.lda_stage.init_random(init_params.get('stdev',0.1))
            else:
                self.lda_stage.init_with_lda(x, speaker_ids, init_params, sec_ids=domain_ids)

            x2_torch = self.lda_stage(x_torch)
            x2 = x2_torch.cpu().numpy()

            if hasattr(self,'si_stage1'):
                if self.si_input == 'main_input':
                    si_input_torch = x_torch
                    si_input = x
                else:
                    si_input_torch = x2_torch
                    si_input = x2

                if init_params.get("random"):
                    self.si_stage1.init_random(init_params.get('stdev',0.1))
                else:
                    self.si_stage1.init_with_lda(si_input, speaker_ids, init_params, sec_ids=domain_ids, complement=True)

                s2_torch = self.si_stage1(si_input_torch)
                    
                if init_params.get('init_si_stage2_with_domain_gb', False):
                    # Initialize the second stage of the si-extractor to be a gaussian backend that predicts
                    # the posterior of each domain. In this case, the number of domains has to coincide with the
                    # dimension of the side info vector
                    assert self.si_dim == len(np.unique(domain_ids))
                    self.si_stage2.init_with_lda(s2_torch.cpu().numpy(), domain_ids, init_params, sec_ids=speaker_ids, gaussian_backend=True)
    
                else:
                    # This is the only component that is initialized randomly unless otherwise indicated by the variable "init_si_stage2_with_domain_gb"
                    self.si_stage2.init_random(init_params.get('w_init', 0.5), init_params.get('b_init', 0.0), init_params.get('type', 'normal'))

                if hasattr(self,'shift_selector'):
                    # Initialize the shifts as the mean of the lda outputs weighted by the si
                    si_torch = self.si_stage2(s2_torch)
                    si = si_torch.cpu().numpy()
                    if init_params.get("random"):
                        self.shift_selector.init_random(init_params.get('stdev',0.1))
                    else:
                        self.shift_selector.init_with_weighted_means(x2, si)
                    x2_torch -= self.shift_selector(si_torch)
                    x2 = x2_torch.cpu().numpy()

            if init_params.get("random"):
                self.plda_stage.init_random(init_params.get('stdev',0.1))
                if self.enrollment is not None:
                    self.enrollment.init_random(init_params.get('stdev',0.1))
            else:    
                self.plda_stage.init_with_plda(x2, speaker_ids, init_params, domain_ids=domain_ids)
                if self.enrollment is not None:
                    # idx_to_str['speaker_id_inv'] contains a mapping from string to indices corresponding to the speaker_ids array
                    weights = compute_weights(speaker_ids, domain_ids, init_params.get('balance_by_domain'))
                    w = weights['sample_weights'][:,np.newaxis]
                    speaker_ids_onehot = utils.onehot_for_class_ids(speaker_ids, self.enrollment_classes, idx_to_str['speaker_id_inv'])
                    self.enrollment.init_with_weighted_means(x2, speaker_ids_onehot * w)

            # Since the training data is usually large, we cannot create all possible trials for x3.
            # So, to create a bunch of trials, we just create a trial loader with a large batch size.
            # This means we need to rerun the front stage and lda again, but it is a small price to pay for the 
            # convenience of reusing the machinery of trial creation in the TrialLoader.
            loader = ddata.TrialLoader(dataset, device, seed=0, batch_size=2000, num_batches=1, balance_by_domain=balance_by_domain, subset=subset)
            x_torch, meta_batch = next(loader.__iter__())
            if hasattr(self,'front_stage'):
                x_torch = self.front_stage(x_torch)
                x = x_torch.cpu().numpy()
            x2_torch = self.lda_stage(x_torch)
            scrs_torch = self.plda_stage(x2_torch)
            same_spk_torch, valid_torch = utils.create_scoring_masks(meta_batch)
            scrs, same_spk, valid = [v.detach().cpu().numpy() for v in [scrs_torch, same_spk_torch, valid_torch]]

            if init_params.get("random"):
                self.cal_stage.init_random(init_params.get('stdev',0.1))
            else:
                self.cal_stage.init_with_logreg(scrs, same_spk, valid, config.ptar, std_for_mats=init_params.get('std_for_cal_matrices',0))

            dummy_durs = torch.ones(scrs.shape[0]).to(device) 
            dummy_si = torch.zeros(scrs.shape[0], self.si_dim).to(device)
            llrs_torch = self.cal_stage(scrs_torch, dummy_durs, dummy_si)
            mask = np.ones_like(same_spk, dtype=int)
            mask[~same_spk] = -1
            mask[~valid] = 0
            
            return compute_loss(llrs_torch, mask=utils.np_to_torch(mask, device), ptar=config.ptar, loss_type=config.loss)


class CA_Calibrator(nn.Module):
    def __init__(self, config, si_dim=None):
        """
        Implements the calibration stage. There are four options:
        * Plain calibration with a global alpha and beta
        * Duration-dependent calibration
        * Side-info- and duration-dependent calibration
        * Side-info-dependent calibration
        """
        super().__init__()

        self.config = config
        self.is_sidep = False
        self.is_durdep = False

        if self.config:
            self.dur_rep = self.config.get('duration_representation', 'log') 
            self.dur_thresholds = self.config.get('duration_thresholds')
            self.si_cal_first = self.config.get('si_cal_first', True) 
            numdurf = self._process_durs(torch.ones(10), init=True).shape[1]

        if self.config and self.config.get('sideinfo_dependent'):
            self.is_sidep = True
            use_G = config.get("use_G_for_si", True) 
            use_L = config.get("use_L_for_si", True) 
            use_C = config.get("use_C_for_si", True)
            self.si_dim = si_dim
            if self.config.get('concat_durfeats_with_sideinfo'):
                self.si_dim += numdurf
            self.sidep_alpha = Quadratic(self.si_dim, use_G=use_G, use_C=use_C, use_L=use_L)
            self.sidep_beta  = Quadratic(self.si_dim, use_G=use_G, use_C=use_C, use_L=use_L)

        if self.config and self.config.get('duration_dependent'):
            self.is_durdep = True
            # Fake input to _process_durs just to get the dimension of the processed duration vector
            use_G = config.get("use_G_for_dur", True) 
            use_L = config.get("use_L_for_dur", True) 
            use_C = config.get("use_C_for_dur", True) 
            self.durdep_alpha = Quadratic(numdurf, use_G=use_G, use_C=use_C, use_L=use_L)
            self.durdep_beta  = Quadratic(numdurf, use_G=use_G, use_C=use_C, use_L=use_L)
        
        if not self.is_sidep and not self.is_durdep:
            # Global calibration
            self.alpha = Parameter(torch.tensor(1.0))
            self.beta  = Parameter(torch.tensor(0.0))


    def forward(self, scores, durations=None, side_info=None):
        # The forward method assumes the same enroll and test data. This is used in training.
        return self.calibrate(scores, durations, side_info)

    @staticmethod
    def _bucketize(tensor, boundaries):
        # Can be replaced with torch.bucketize in older versions
        result = torch.zeros_like(tensor, dtype=torch.int64)
        for boundary in boundaries:
            result += (tensor > boundary).int()
        return result

    def _process_durs(self, durations, init=False):
        
        if self.dur_rep == 'log':
            durs = torch.log(durations.unsqueeze(1))

        elif self.dur_rep == 'logpa':
            # Logarithm with a shift initiliazed with 0
            if init: 
                self.logdur_shift = Parameter(torch.tensor(0.0))
            durs = torch.log(durations.unsqueeze(1)+self.logdur_shift)

        elif self.dur_rep == 'log2':
            durs = torch.log(durations.unsqueeze(1))
            durs = torch.cat([durs, durs*durs], 1)

        elif self.dur_rep == 'idlog':
            durs = durations.unsqueeze(1)
            durs = torch.cat([durs/100, torch.log(durs)], 1)

        elif self.dur_rep == 'siglog' or self.dur_rep == 'siglogp':
            durs = torch.log(durations.unsqueeze(1))
            # Take the log durs and multiplied them by a sigmoid centered a some value
            # and then again with a rotated sigmoid
            if init:
                scale   = self.config.get("duration_sigmoid_scale")
                centers = np.log(self.dur_thresholds)
                if self.dur_rep == 'siglogp':
                    # Scale and centers are learnable parameters, initialized
                    # with the provided value
                    self.siglogdur_scale   = Parameter(torch.tensor(scale))
                    self.siglogdur_centers = Parameter(torch.tensor(centers))
                else:
                    self.siglogdur_scale   = scale
                    self.siglogdur_centers = centers

            fs = []
            prev_sig = 0
            for c in self.siglogdur_centers:
                sigf = torch.sigmoid(-self.siglogdur_scale*(durs-c)) 
                sig = sigf - prev_sig
                fs.append(durs*sig)
                prev_sig = sigf
            sig = 1-sigf
            fs.append(durs*sig)
            durs = torch.cat(fs, 1)
            # Simple version for a single center. 
            #    sig1 = torch.sigmoid(scale*(durs-center))
            #sig2 = 1.0-sig1
            #durs = torch.cat([durs*sig1, durs*sig2], 1)

        else:
            durs_disc = self._bucketize(durations, self.dur_thresholds)
            durs_onehot = nn.functional.one_hot(durs_disc, num_classes=len(self.dur_thresholds)+1).type(torch.get_default_dtype())
            if self.dur_rep == 'pwlog':
                durs = torch.log(durations.unsqueeze(1)) * durs_onehot
            elif self.dur_rep == 'discrete':            
                durs = durs_onehot
            else:
                raise Exception("Duration representation %s not implemented"%self.dur_rep)

        return durs

    def calibrate(self, scores, durations_test=None, side_info_test=None, durations_enroll=None, side_info_enroll=None):

        if self.is_sidep:
            # Sideinfo-dep calibration, optionally with the duration appended
            if self.config.get('concat_durfeats_with_sideinfo'):
                durs_enroll = self._process_durs(durations_enroll) if durations_enroll is not None else None
                durs_test   = self._process_durs(durations_test) 
                side_info_enroll = torch.cat([durs_enroll, side_info_enroll], 1) if side_info_enroll is not None else None
                side_info_test   = torch.cat([durs_test,   side_info_test],   1) 
            alpha_si = self.sidep_alpha.score(side_info_test, side_info_enroll)
            beta_si  = self.sidep_beta.score(side_info_test, side_info_enroll)
            if self.si_cal_first:
                scores = alpha_si*scores + beta_si

        if self.is_durdep:
            # Dur-dep calibration 
            durs_enroll = self._process_durs(durations_enroll) if durations_enroll is not None else None
            durs_test   = self._process_durs(durations_test) 
            alpha = self.durdep_alpha.score(durs_test, durs_enroll)
            beta  = self.durdep_beta.score(durs_test, durs_enroll)

        elif not self.is_sidep:
            # Global calibration
            alpha = self.alpha
            beta  = self.beta
        
        else:
            # Sideinfo-dep calibration only
            alpha = 1.0
            beta = 0.0

        scores = alpha*scores + beta

        if self.is_sidep and not self.si_cal_first:
            scores = alpha_si*scores + beta_si

        return scores


    def init_random(self, std):

        if self.is_durdep:
            # Dur-dep calibration (might also have an si-dependent stage)
            self.durdep_alpha.init_random(std)
            self.durdep_beta.init_random(std)

        elif not self.is_sidep:
            # Global calibration
            nn.init.normal_(self.alpha, 0.0, std)
            nn.init.normal_(self.beta,  0.0, std)
        else:
            self.sidep_alpha.init_random(std)
            self.sidep_beta.init_random(std)


    def init_with_logreg(self, scores, same_spk, valid, ptar, std_for_mats=0):

        scores = scores[valid]
        labels = same_spk[valid]

        tar = scores[labels==1]
        non = scores[labels==0]

        a, b = calibration.logregCal(tar, non, ptar, return_params=True)

        if self.is_durdep:
            # Dur-dep calibration (might also have an si-dependent stage)
            self.durdep_alpha.init_with_constant(a, std_for_mats)
            self.durdep_beta.init_with_constant(b, std_for_mats)
            if self.is_sidep:
                self.sidep_alpha.init_with_constant(1.0, std_for_mats)
                self.sidep_beta.init_with_constant(0.0, std_for_mats)

        elif not self.is_sidep:
            # Global calibration
            utils.replace_state_dict(self, {'alpha': a, 'beta': b})

        else:
            # Only si-dependent cal
            self.sidep_alpha.init_with_constant(a, std_for_mats)
            self.sidep_beta.init_with_constant(b, std_for_mats)


class Quadratic(nn.Module):
    """ Note that, while I am calling this class Quadratic, it is actually a
    second order polynomial on the inputs rather than just quadratic"""
    def __init__(self, in_dim, use_G=True, use_C=True, use_L=True):
        super().__init__()
        self.L = Parameter(torch.zeros(in_dim, in_dim)) if use_L else None
        self.G = Parameter(torch.zeros(in_dim, in_dim)) if use_G else None
        self.C = Parameter(torch.zeros(in_dim, 1)) if use_C else None
        self.k = Parameter(torch.tensor(0.0))

    def forward(self, x):
        # The forward method assumes the same enroll and test data. This is used in training.
        return self.score(x)

    def score(self, xt, xe=None):
        # PLDA-like computation. The output i, j is given by
        # 2 * xi^T L xj + xi^T G xi + xj^T G xj + xi^T C + xj^T C + k
        if xe is None:
            symmetric = True
            xe = xt
        else:
            symmetric = False

        if self.L is not None:
            Lterm = torch.matmul(torch.matmul(xe, 0.5*(self.L+self.L.T)), xt.T)
        else:   
            Lterm = torch.zeros(xe.shape[0],xt.shape[0]) 
        
        if self.C is not None:
            Ctermt = torch.matmul(xt, self.C) 
            Cterme = torch.matmul(xe, self.C) if not symmetric else Ctermt
        else:   
            Ctermt = torch.zeros(xt.shape[0],1) 
            Cterme = torch.zeros(xe.shape[0],1) if not symmetric else Ctermt

        if self.G is not None:
            Gtermt = torch.sum(xt * torch.matmul(xt, 0.5*(self.G+self.G.T)) , 1, keepdim=True) 
            Gterme = torch.sum(xe * torch.matmul(xe, 0.5*(self.G+self.G.T)) , 1, keepdim=True) if not symmetric else Gtermt 
        else:
            Gtermt = torch.zeros(xt.shape[0],1)
            Gterme = torch.zeros(xe.shape[0],1) if not symmetric else Gterme
        
        output = 2 * Lterm + Gterme + Gtermt.T + Cterme + Ctermt.T + self.k

        return output


    def init_with_constant(self, k, std_for_mats=0):
        # Initialize k with a given value and, optionally, the 
        # L G and C matrices with a normal
        if std_for_mats>0:
            self.init_random(std_for_mats)
        utils.replace_state_dict(self, {'k': k})

    def init_random(self, std):
        nn.init.normal_(self.k, 0.0, std)
        nn.init.normal_(self.L, 0.0, std)
        nn.init.normal_(self.G, 0.0, std)
        nn.init.normal_(self.C, 0.0, std)

    def init_with_plda(self, x, speaker_ids, init_params, domain_ids=None):
        # Compute a PLDA model with the input data, approximating the 
        # model with just the usual initialization without the EM iterations

        weights = compute_weights(speaker_ids, domain_ids, init_params.get('balance_by_domain'))

        # Debugging of weight usage in PLDA:
        # Repeat the data from the first 5000 speakers twice either explicitely or through the weights
        # These two models should be identical (and they are!)
        #sela = speaker_ids<5000
        #selb = speaker_ids>=5000
        #x2 = np.concatenate([x[sela],x[sela],x[selb]])
        #speaker_ids2 = np.concatenate((speaker_ids[sela], speaker_ids[sela]+np.max(speaker_ids)+1, speaker_ids[selb]))
        #weights2 = np.ones(len(np.unique(speaker_ids2)))
        #BCov2, WCov2, mu2 = compute_2cov_plda_model(x2, speaker_ids2, weights2, 10)
        #weights3 = weights.copy()
        #weights3[0:5000] *= 2
        #BCov3, WCov3, mu3 = compute_2cov_plda_model(x, speaker_ids, weights3, 10)
        #assert np.allclose(BCov2,BCov3)
        #assert np.allclose(WCov2,WCov3)
        #assert np.allclose(mu2, mu3)

        # Use the cross-product between speaker and domain ids for initializing LDA and PLDA
        # This can be used to be able to increase the LDA dimension beyond the number of available
        # speakers (or languages)
        if init_params.get('lda_by_class_x_domain'):
            # Replace the class_ids with new indices that correspond to the cross-product between
            # class_ids and sec_ids. In this case, the balance_by_domain option does not make sense
            assert init_params.get('balance_by_domain') is not True
            _, init_ids = np.unique([str(s)+','+str(c) for s,c in zip(speaker_ids, domain_ids)], return_inverse=True)
            # Need to recompute the weights (which will all be 1.0 since balance_by_domain is false), since
            # we now have more classes
            weights = compute_weights(init_ids, domain_ids, init_params.get('balance_by_domain'))
        else:
            init_ids = None

        # Bi and Wi are the between and within covariance matrices and mu is the global (weighted) mean
        Bi, Wi, mu = compute_2cov_plda_model(x, speaker_ids, weights, init_params.get('plda_em_its',0), init_ids=init_ids)
        
        # Equations (14) and (16) in Cumani's paper 
        # Note that the paper has an error in the formula for k (a 1/2 missing before k_tilde) 
        # that is fixed in the equations below
        # To compute L_tild and G_tilde we use the following equality:
        # inv( inv(C) + n*inv(D) ) == C @ inv(D + n*C) @ D == C @ solve(D + n*C, D)
        B              = utils.CholInv(Bi)
        W              = utils.CholInv(Wi)
        Bmu            = B @ mu.T
        L_tilde        = Bi @ np.linalg.solve(Wi + 2*Bi, Wi)
        G_tilde        = Bi @ np.linalg.solve(Wi + Bi, Wi)
        WtGL           = W @ (L_tilde-G_tilde)
        logdet_L_tilde = np.linalg.slogdet(L_tilde)[1]
        logdet_G_tilde = np.linalg.slogdet(G_tilde)[1]
        logdet_B       = B.logdet()       
        k_tilde        = -2.0*logdet_G_tilde + logdet_L_tilde - logdet_B + mu @ Bmu

        k = 0.5 * k_tilde + 0.5 * Bmu.T @ (L_tilde - 2*G_tilde) @ Bmu
        L = 0.5 * (W @ (W @ L_tilde)).T
        G = 0.5 * (W @ WtGL).T
        C = (WtGL @ Bmu)

        state_dict = {'L': L, 'G': G, 'C': C, 'k': k.squeeze()}

        utils.replace_state_dict(self, state_dict)



class Affine(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None, bias=True):
    
        super().__init__()
        self.activation = activation
        self.W = Parameter(torch.zeros(in_dim, out_dim))
        self.has_bias = bias
        if bias:
            self.b = Parameter(torch.zeros(out_dim))
        else:
            self.b = 0

    def forward(self, x):
        preact = torch.matmul(x, self.W) + self.b

        if self.activation == "l2norm":
            out = preact/preact.norm(dim=1, keepdim=True)
        elif self.activation == "logsoftmax":
            out = nn.functional.log_softmax(preact, dim=1)
        elif self.activation == "softmax":
            out = nn.functional.softmax(preact, dim=1)
        elif self.activation == "sigmoid":
            out = nn.functional.sigmoid(preact)
        elif self.activation == "relu":
            out = nn.functional.relu(preact)
        elif self.activation == "none" or self.activation is None:
            out = preact
        else:
            raise Exception("Activation %s not implemented"%self.activation)
        return out

    def init_with_params(self, param_dict):
        utils.replace_state_dict(self, param_dict)

    def init_with_lda(self, x, class_ids, init_params, complement=True, sec_ids=None, gaussian_backend=False):

        if self.has_bias is False:
            raise Exception("Cannot initialize this component with init_with_lda because it was created with bias=False")

        if init_params.get('lda_by_class_x_domain'):
            # Replace the class_ids with new indices that correspond to the cross-product between
            # class_ids and sec_ids. In this case, the balance_by_domain option does not make sense
            assert init_params.get('balance_by_domain') is not True
            _, class_ids = np.unique([str(s)+','+str(c) for s,c in zip(class_ids, sec_ids)], return_inverse=True)

        weights = compute_weights(class_ids, sec_ids, init_params.get('balance_by_domain'))

        BCov, WCov, GCov, mu, mu_per_class, _ = compute_lda_model(x, class_ids, weights)

        if gaussian_backend:
            W = linalg.solve(WCov, mu_per_class.T, sym_pos=True)
            b = - 0.5 * (W * mu_per_class.T).sum(0)

        else:
            evals, evecs = linalg.eigh(BCov, WCov)
            evecs = evecs[:, np.argsort(evals)[::-1]] 

            lda_dim = self.W.shape[1]
            if complement:
                W = evecs[:,:lda_dim]
            else:
                W = evecs[:,-lda_dim:]

            # Normalize each dimension so that output features will have variance 1.0
            if init_params.get('variance_norm_lda', True):
                W = W @ (np.diag(1. / np.sqrt(np.diag(W.T @ GCov @ W))))

            # Finally, estimate the shift so that output features have mean 0.0 
            mu = mu @ W
            b = -mu

        utils.replace_state_dict(self, {'W': W, 'b': b})

    def init_random(self, W_param, b_param=None, init_type="normal"):

        if b_param is None:
            b_param = W_param

        if init_type == 'normal':
            nn.init.normal_(self.W, 0.0, W_param)

            # If b_param is 0 it means we want to initialize b with a vector of 0s
            if b_param > 0:
                nn.init.normal_(self.b, 0.0, b_param)
            else:
                nn.init.constant_(self.b, 0.0)
        else:
            raise Exception("Initialization type %s not implement"%init_type)


    def init_with_weighted_means(self, data, weights):

        Wl = list()
        for w in weights.T:
            Wl.append(np.dot(data.T, w)/np.sum(w))
        utils.replace_state_dict(self, {'W': np.c_[Wl].T})


class SimpleNN(nn.Module):

    def __init__(self, in_dim, out_dim, out_activation, inner_sizes, inner_activation):
        super().__init__()

        self.hidden = nn.ModuleList()
        in_dimi = in_dim 
        for size in inner_sizes:
            self.hidden.append(Affine(in_dimi, size, inner_activation))
            in_dimi = size

        self.outlayer = Affine(size, out_dim, out_activation)


    def forward(self, x):
        for layer in self.hidden:
            x = layer(x)
        return self.outlayer(x)


    def init_random(self, W_param, b_param, init_type):

        for layer in self.hidden:
            layer.init_random(W_param, b_param, init_type)

        self.outlayer.init_random(W_param, b_param, init_type)



def compute_2cov_plda_model(x, class_ids, class_weights, em_its=0, init_ids=None):
    """ Follows the "EM for SPLDA" document from Niko Brummer:
    https://sites.google.com/site/nikobrummer/EMforSPLDA.pdf
    """

    if init_ids is None:
        init_ids = class_ids

    BCov, WCov, GCov, mu, muc, stats = compute_lda_model(x, init_ids, class_weights)
    W = utils.CholInv(WCov)
    v, e = linalg.eig(BCov)
    V = e * np.sqrt(np.real(v))
    
    def mstep(R, T, S, n_samples):
        V = (utils.CholInv(R) @  T).T
        Winv = 1/n_samples * (S-V@T)
        W = utils.CholInv(Winv)
        return V, W, Winv

    def estep(stats, V, W):
        VtW  = (W @ V).T
        VtWV = VtW @ V
        VtWf = VtW @ stats.F.T
        y_hat = np.zeros_like(stats.F).T
        R = np.zeros_like(V)
        llk = 0.0
        for n in np.unique(stats.N):
            idxs = np.where(stats.N == n)[0]
            L = n * VtWV + np.eye(V.shape[0])
            Linv = utils.CholInv(L)
            y_hat[:,idxs] = Linv @ VtWf[:,idxs]
            # The expression below is a robust way for solving 
            # yy_sum = len(idxs)*Linv + np.dot(y_hat[:, idxs], y_hat[:, idxs].T)
            # while avoiding doing the inverse of L which can create numerical issues
            n_spkrs = np.sum(stats.cweights[idxs])
            yy_sum = np.linalg.solve(L, n_spkrs * np.eye(V.shape[0]) + L @ y_hat[:, idxs] @ (y_hat[:, idxs].T * stats.cweights[idxs]))
            R += n*yy_sum
            llk += 0.5 * n_spkrs * Linv.logdet()
        T = y_hat @ (stats.F * stats.cweights)
        llk += 0.5*np.trace(T @ VtW.T) + 0.5 * np.sum(stats.N*stats.cweights) * W.logdet() - 0.5 * np.trace(W @ stats.S)
        return R, T, llk/np.sum(stats.N*stats.cweights)

    prev_llk = 0.0
    for it in range(em_its):
        R, T, llk = estep(stats, V, W)
        V, W, WCov = mstep(R, T, stats.S, np.sum(stats.N*stats.cweights))
        print("EM it %d LLK = %.5f (+%.5f)" % (it, llk, llk-prev_llk))
        prev_llk = llk 
    
    BCov = V @ V.T

    return BCov, WCov, np.atleast_2d(mu)

def compute_stats(x, class_ids, class_weights, sample_weights):
    # Zeroth, first and second order stats per speaker, considering that
    # each speaker is weighted by the provided weights
    
    stats     = utils.AttrDict(dict())
    nsamples  = class_ids.shape[0]
    classmat  = coo_matrix((np.ones(nsamples), (class_ids, np.arange(nsamples)))).tocsr()
    stats.cweights = np.atleast_2d(class_weights).T
    stats.sweights = np.atleast_2d(sample_weights).T / np.sum(sample_weights) * float(nsamples)
    sweightsdiag   = dia_matrix((stats.sweights.squeeze(),0),shape=(nsamples,nsamples))
    classmat_sw    = classmat.dot(sweightsdiag) # Matrix with one row per speaker and one col per sample, with the weights as entries

    # Global mean, considering that each speaker's data should be 
    # multiplied by the corresponding weight for that speaker
    sample_weights = stats.cweights[class_ids] * stats.sweights
    stats.mu       = np.array(sample_weights.T @ x) / np.sum(sample_weights) 

    # N and F are stats per speaker, weights are not involved in the computation of those. 
    # S, on the other hand, is already a sum over speakers so the sample from each speaker
    # has to be weighted by the corresponding weight
    xcent     = x - stats.mu
    xcentw    = xcent * sample_weights
    stats.N   = np.array(classmat_sw.sum(1)) 
    stats.F   = np.array(classmat_sw @ xcent)
    stats.S   = xcent.T @ xcentw

    return stats


def compute_lda_model(x, class_ids, weights):

    class_weights = weights['speaker_weights']
    sample_weights = weights['sample_weights']

    # Simpler code using sklearn. We are not using this because it does not allow for weights
    # lda = discriminant_analysis.LinearDiscriminantAnalysis(store_covariance=True)
    # lda.fit(x, class_ids)
    # WCov = lda.covariance_
    # BCov = np.cov(x.T, bias=1) - WCov
    # mu = np.mean(x, axis=0)
    # GCov should be the same as np.cov(x.T,  bias=1)

    stats = compute_stats(x, class_ids, class_weights, sample_weights)

    # Once we have the stats, we do not need x or class_ids any more (we still do need the class weights)
    mu    = stats.mu
    muc   = stats.F / stats.N + stats.mu # Means by class (not centered)
    
    # Matrix with the weights per speaker in the diagonal
    n_spkrs = stats.N.shape[0]
    weightsdia = dia_matrix((class_weights,0),shape=(n_spkrs, n_spkrs))

    # BCov and GCov are computed by weighting the contribution of each speaker with 
    # the provided weights
    Ntot  = np.sum(np.multiply(stats.N, stats.cweights))
    Fs    = stats.F / np.sqrt(stats.N) 
    BCov  = Fs.T @ (Fs * stats.cweights) / Ntot
    GCov  = stats.S / Ntot
    WCov  = GCov - BCov

    return BCov, WCov, GCov, mu.squeeze(), muc, stats



def compute_weights(speaker_ids, domain_ids, balance_by_domain=False):
    
    spk_weights = None
    sample_weights = None

    if balance_by_domain == 'num_classes_per_dom_prop_to_its_total_num_classes':
        # Assign one weight per sample, so that all classes in each domain and
        # accross domains weight the same. That is, each sample is weighted by
        # the inverse of the number of samples for that language and domain.
        # If all classes appear in a single domain, this is very similar to
        # the per-speaker implementation of the weights below.
        # This can also be used to balance out the language-id samples so that all 
        # languages have the same equivalent number of samples for each domain they 
        # appear in.
        _, unique_sids_x_doms = np.unique([str(s)+','+str(c) for s,c in zip(speaker_ids, domain_ids)], return_inverse=True)
        weight_per_sid_x_dom = 1.0/np.bincount(unique_sids_x_doms)
        weight_per_sid_x_dom *= len(weight_per_sid_x_dom)/np.sum(weight_per_sid_x_dom)
        sample_weight =  weight_per_sid_x_dom[unique_sids_x_doms]
    else:
        sample_weight = np.ones_like(speaker_ids, dtype=float)

    unique_sids = np.unique(speaker_ids)
    unique_doms = np.unique(domain_ids)
    dom_weight = np.ones_like(unique_doms, dtype=float)
    for d in unique_doms:
        if balance_by_domain is True or balance_by_domain == 'same_num_classes_per_dom':
            # The weight for each domain is given by the inverse of the number of speakers for that domain    
            # If all domains have the same number of speakers, then all weights are 1.0
            dom_weight[d] = len(unique_sids)*1.0/len(np.unique(speaker_ids[domain_ids==d]))/len(unique_doms)
        else:
            # Dummy weight of 1 for all domains
            dom_weight[d] = 1.0

    # The weight for each speaker is given by the weight for the domain to which it belongs.
    # This assumes each speaker belongs to a single domain, which will mostly be the case.
    # Else, we'll just assign a weight that corresponds to one of its domains.
    spk_weight = np.ones_like(unique_sids, dtype=float)
    for s in unique_sids:
        spk_weight[s] = dom_weight[domain_ids[np.where(speaker_ids==s)[0][0]]]

    print("Weights per domain: %s"%dom_weight)

    return {'speaker_weights': spk_weight, 'sample_weights': sample_weight}

