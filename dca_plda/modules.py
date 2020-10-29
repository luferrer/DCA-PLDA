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

def compute_loss(llrs, metadata=None, ptar=0.01, mask=None, loss_type='cross_entropy', return_info=False):

    if metadata is None:
        assert mask is not None
        # The mask is assumed to have been generated using the Key class, 
        # so targets are labeled with 1, impostors with -1 and non-scored trials with 0.
        valid = mask != 0
        same_spk = mask == 1
    else:
        assert metadata is not None
        same_spk, valid = utils.create_scoring_masks(metadata)

    # Shift the llrs to convert them to posteriors
    logits = llrs + logit(ptar)
    logits = logits[valid]
    labels = same_spk[valid]
    labels = labels.type(logits.type())

    # The loss will be given by tar_weight * tar_loss + imp_weight * imp_loss
    ptart = torch.as_tensor(ptar)
    tar_weight = ptart/torch.sum(labels==1)
    imp_weight = (1-ptart)/torch.sum(labels==0)

    # Finally, compute the loss and multiply it by the weight that corresponds to the impostors
    # Loss types are taken from Niko Brummer's paper: "Likelihood-ratio calibration using prior-weighted proper scoring rules"
    if loss_type == "cross_entropy":
        criterion = nn.BCEWithLogitsLoss(pos_weight=tar_weight/imp_weight, reduction='sum')
        baseline_loss = -ptar*np.log(ptar) - (1-ptar)*np.log(1-ptar)
        loss = criterion(logits, labels)*imp_weight/baseline_loss
    elif loss_type == "brier":
        baseline_loss = ptar * (1-ptar)**2 + (1-ptar) * ptar**2
        posteriors = torch.sigmoid(logits)
        loss = torch.sum(labels*tar_weight*(1-posteriors)**2 + (1-labels)*imp_weight*posteriors**2)/baseline_loss

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

        self.lda_stage  = Affine(in_dim, lda_dim, "l2norm")

        self.plda_stage = Quadratic(lda_dim)

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
        return self.score(x, dure=durations)


    def score(self, xe, xt=None, dure=None, durt=None, si_only=False):
        # Same as the forward method but it allows for assymetric scoring where the rows and columns
        # of the resulting score file corresponds to two different sets of data

        hast = xt is not None

        x2e = self.lda_stage(xe)
        x2t = self.lda_stage(xt) if hast else None
        
        if hasattr(self,'si_stage1'):
            si_inpute = xe if self.si_input == 'main_input' else x2e
            si_inputt = xt if self.si_input == 'main_input' else x2t
             
            s2e = self.si_stage1(si_inpute)
            s2t = self.si_stage1(si_inputt) if hast else None

            sie = self.si_stage2(s2e)
            sit = self.si_stage2(s2t) if hast else None

        else:
            sie = sit = None

        if si_only:
            assert xt is None and sie is not None
            return sie
        else:

            if self.config.get('si_dependent_shift_parameters'):
                # Use the si to find a shift vector for each sample 
                x2e = x2e - self.shift_selector(sie)
                x2t = x2t - self.shift_selector(sit) if hast else None

            scrs = self.plda_stage.score(x2e, x2t)
            llrs = self.cal_stage.calibrate(scrs, dure, sie, durt, sit)            
            return llrs


    def init_params_with_data(self, dataset, config, device=None, subset=None):

        balance_by_domain = config.balance_batches_by_domain
        assert 'init_params' in config
        init_params = config.init_params
        
        # The code here repeats the steps in forward above, but adds the steps necessary for initialization.
        # I chose to keep these two methods separate to leave the forward small and easy to read.
        with torch.no_grad():

            x, meta, _ = dataset.get_data_and_meta(subset)
            speaker_ids = meta['speaker_id']
            domain_ids  = meta['domain_id']
            x_torch = utils.np_to_torch(x, device)
            
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

                self.si_stage1.init_with_lda(si_input, speaker_ids, init_params, sec_ids=domain_ids, complement=True)
                s2_torch = self.si_stage1(si_input_torch)
                    
                if init_params.get('init_si_stage2_with_domain_gb', False):
                    # Initialize the second stage of the si-extractor to be a gaussian backend that predicts
                    # the posterior of each domain. In this case, the number of domains has to coincide with the
                    # dimension of the side info vector
                    assert self.si_dim == len(np.unique(domain_ids))
                    self.si_stage2.init_with_lda(s2_torch.cpu().numpy(), domain_ids, init_params, sec_ids=speaker_ids, gaussian_backend=True)
    
                else:
                    self.si_stage2.init_random(init_params.get('w_init', 0.5), init_params.get('b_init', 0.0), init_params.get('type', 'normal'))

                if hasattr(self,'shift_selector'):
                    # Initialize the shifts as the mean of the lda outputs weighted by the si
                    si_torch = self.si_stage2(s2_torch)
                    si = si_torch.cpu().numpy()
                    self.shift_selector.init_with_weighted_means(x2, si)
                    x2_torch -= self.shift_selector(si_torch)
                    x2 = x2_torch.cpu().numpy()

            self.plda_stage.init_with_plda(x2, speaker_ids, init_params, domain_ids=domain_ids)

            # Since the training data is usually large, we cannot create all possible trials for x3.
            # So, to create a bunch of trials, we just create a trial loader with a large batch size.
            # This means we need to rerun lda again, but it is a small price to pay for the 
            # convenience of reusing the machinery of trial creation in the TrialLoader.
            loader = ddata.TrialLoader(dataset, device, seed=0, batch_size=2000, num_batches=1, balance_by_domain=balance_by_domain, subset=subset)
            x_torch, meta_batch = next(loader.__iter__())
            x2_torch = self.lda_stage(x_torch)
            scrs_torch = self.plda_stage(x2_torch)
            same_spk_torch, valid_torch = utils.create_scoring_masks(meta_batch)
            scrs, same_spk, valid = [v.detach().cpu().numpy() for v in [scrs_torch, same_spk_torch, valid_torch]]
        
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

        if self.config and self.config.get('sideinfo_dependent'):
            assert config is not None
            self.is_sidep = True
            self.si_dim = si_dim
            self.si_cal_first = config.get('si_cal_first', True) if config is not None else True
            use_G = config.get("use_G_for_si", config.get("use_G",True)) 
            use_C = config.get("use_C_for_si", config.get("use_C",True))
            self.sidep_alpha = Quadratic(self.si_dim, use_G=use_G, use_C=use_C)
            self.sidep_beta  = Quadratic(self.si_dim, use_G=use_G, use_C=use_C)

        if self.config and self.config.get('duration_dependent'):
            self.is_durdep = True
            self.dur_rep = self.config.get('duration_representation', 'log') 
            if self.dur_rep == 'log':
                numdurf = 1
            elif self.dur_rep == 'discrete':
                self.dur_thresholds = self.config.get('duration_thresholds')
                numdurf = len(self.dur_thresholds)+1 
            else:
                raise Exception("Duration representation %s not implemented"%self.dur_rep)
            use_G = config.get("use_G_for_dur", True) 
            use_C = config.get("use_C_for_dur", True) 
            self.durdep_alpha = Quadratic(numdurf, use_G=use_G, use_C=use_C)
            self.durdep_beta  = Quadratic(numdurf, use_G=use_G, use_C=use_C)
        
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

    def _process_durs(self, durations):
        
        if self.dur_rep == 'log':
            durs = durations.unsqueeze(1)
            durs = torch.log(durs)
        else:
            durs_disc = self._bucketize(durations, self.dur_thresholds)
            durs = nn.functional.one_hot(durs_disc, num_classes=len(self.dur_thresholds)+1).type(torch.get_default_dtype())

        return durs

    def calibrate(self, scores, durations_enroll=None, side_info_enroll=None, durations_test=None, side_info_test=None):

        if self.is_sidep:
            # Sideinfo-dep calibration
            alpha_si = self.sidep_alpha.score(side_info_enroll, side_info_test)
            beta_si  = self.sidep_beta.score(side_info_enroll, side_info_test)
            if self.si_cal_first:
                scores = alpha_si*scores + beta_si

        if self.is_durdep:
            # Dur-dep calibration 
            durs_enroll = self._process_durs(durations_enroll)
            durs_test   = self._process_durs(durations_test) if durations_test is not None else None
            alpha = self.durdep_alpha.score(durs_enroll, durs_test)
            beta  = self.durdep_beta.score(durs_enroll, durs_test)

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
    def __init__(self, in_dim, use_G=True, use_C=True):
        super().__init__()
        self.L = Parameter(torch.zeros(in_dim, in_dim))
        self.G = Parameter(torch.zeros(in_dim, in_dim)) if use_G else None
        self.C = Parameter(torch.zeros(in_dim, 1)) if use_C else None
        self.k = Parameter(torch.tensor(0.0))

    def forward(self, x):
        # The forward method assumes the same enroll and test data. This is used in training.
        return self.score(x)

    def score(self, xe, xt=None):
        # PLDA-like computation. The output i, j is given by
        # 2 * xi^T L xj + xi^T G xi + xj^T G xj + xi^T C + xj^T C + k
        if xt is None:
            symmetric = True
            xt = xe
        else:
            symmetric = False

        Lterm  = torch.matmul(torch.matmul(xe, 0.5*(self.L+self.L.T)), xt.T)
        
        if self.C is not None:
            Cterme = torch.matmul(xe, self.C) 
            Ctermt = torch.matmul(xt, self.C) if not symmetric else Cterme
        else:   
            Cterme = torch.zeros(xe.shape[0],1)
            Ctermt = torch.zeros(xt.shape[0],1) if not symmetric else Cterme

        if self.G is not None:
            Gterme = torch.sum(xe * torch.matmul(xe, 0.5*(self.G+self.G.T)) , 1, keepdim=True)  
            Gtermt = torch.sum(xt * torch.matmul(xt, 0.5*(self.G+self.G.T)) , 1, keepdim=True) if not symmetric else Gterme
        else:
            Gterme = torch.zeros(xe.shape[0],1)
            Gtermt = torch.zeros(xt.shape[0],1) if not symmetric else Gterme
        
        output = 2 * Lterm + Gterme + Gtermt.T + Cterme + Ctermt.T + self.k

        return output


    def init_with_constant(self, k, std_for_mats=0):
        # Initialize k with a given value and, optionally, the 
        # L G and C matrices with a normal

        utils.replace_state_dict(self, {'k': k})
        if std_for_mats>0:
            nn.init.normal_(self.L, 0.0, std_for_mats)
            nn.init.normal_(self.G, 0.0, std_for_mats)
            nn.init.normal_(self.C, 0.0, std_for_mats)


    def init_with_plda(self, x, speaker_ids, init_params, domain_ids=None):
        # Compute a PLDA model with the input data, approximating the 
        # model with just the usual initialization without the EM iterations

        # Get the within and between-class covariance matrices
        # WCov is the covariance of the noise term in SPLDA and BCov = inverse(V V^t)

        weights = compute_sample_weights(speaker_ids, domain_ids, init_params)
                
        BCov, WCov, _, mu, _ = compute_lda_model(x, speaker_ids, weights)

        Binv = np.linalg.inv(BCov)
        Winv = np.linalg.inv(WCov)

        # Equations (14) and (16) in Cumani's paper 
        # Note that the paper has a couple of errors in the derivations, in the k_tilde formula and in k
        # that are fixed in the equations below
        Bmu               = np.dot(Binv,mu)
        L_tilde           = np.linalg.inv(Binv+2*Winv)
        G_tilde           = np.linalg.inv(Binv+Winv)
        WtGL              = np.dot(Winv.T,L_tilde-G_tilde)
        _, logdet_L_tilde = np.linalg.slogdet(L_tilde)
        _, logdet_G_tilde = np.linalg.slogdet(G_tilde)
        _, logdet_Binv    = np.linalg.slogdet(Binv)
        k_tilde           = -2.0*logdet_G_tilde - logdet_Binv + logdet_L_tilde + np.dot(mu.T, Bmu) 

        k = 0.5 * k_tilde + 0.5 * np.dot(Bmu.T, np.dot(G_tilde - 2*L_tilde, Bmu))
        L = 0.5 * np.dot(Winv.T,np.dot(L_tilde,Winv))
        G = 0.5 * np.dot(WtGL,Winv)
        C = np.dot(WtGL,Bmu)[:,np.newaxis]

        if init_params.get('norm_plda_params'):
            # Divide all params by k so that we get rid of that scale at init
            # It will be compensated by the calibration params anyway. This is to make 
            # the params more comparable across different initializations so that
            # regularization works similarly.
            L /= k
            G /= k
            C /= k
            k = 1.0

        state_dict = {'L': L, 'G': G, 'C': C, 'k': k}

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

    def init_with_lda(self, x, class_ids, init_params, complement=True, sec_ids=None, gaussian_backend=False):

        if self.has_bias is False:
            raise Exception("Cannot initialize this component with init_with_lda because it was created with bias=False")

        weights = compute_sample_weights(class_ids, sec_ids, init_params)
                
        BCov, WCov, GCov, mu, mu_per_class = compute_lda_model(x, class_ids, weights)

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
            W = W.dot(np.diag(1. / np.sqrt(np.diag(W.T.dot(GCov).dot(W)))))

            # Finally, estimate the shift so that output features have mean 0.0 
            mu = np.matmul(mu, W)
            b = -mu

        utils.replace_state_dict(self, {'W': W, 'b': b})

    def init_random(self, W_param, b_param, init_type):

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
        for i in np.arange(self.W.shape[0]):
            Wl.append(np.dot(data.T, weights[:,i])/np.sum(weights[:,i]))
        utils.replace_state_dict(self, {'W': np.c_[Wl]})


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


def compute_lda_model(x, class_ids, weights):

    # Simpler code using sklearn. We are not using this because it does not allow for weights
    # lda = discriminant_analysis.LinearDiscriminantAnalysis(store_covariance=True)
    # lda.fit(x, class_ids)
    # WCov = lda.covariance_
    # BCov = np.cov(x.T, bias=1) - WCov
    # mu = np.mean(x, axis=0)
    # GCov should be the same as np.cov(x.T,  bias=1)
        
    nsamples    = class_ids.shape[0]
    weights    /= np.sum(weights)/float(nsamples)
    weightsdiag = dia_matrix((weights,0),shape=(nsamples,nsamples))
    classmat    = coo_matrix((np.ones(nsamples), (class_ids, np.arange(nsamples)))).tocsr()
    classmatw   = classmat.dot(weightsdiag) # Matrix with one row per speaker and one col per sample, with the weights as entries
    mu          = np.array(weights.dot(x)) / float(nsamples) # Global mean
    counts      = np.array(classmatw.sum(1)) # Counts by class
    muc         = np.array(classmatw.dot(x)) / counts # Means by class
    mucCS       = (muc - mu) * np.sqrt(counts) # Means by class, centered and scaled
    BCov        = mucCS.T.dot(mucCS) / nsamples # Between class covariance
    xCS         = np.multiply((x - mu).T, np.sqrt(weights)).T # Samples centered and scaled
    GCov        = np.dot(xCS.T,xCS) / nsamples
    WCov        = GCov - BCov

    return BCov, WCov, GCov, mu, muc


def compute_sample_weights(speaker_ids, domain_ids, init_params):

    weights = np.ones_like(speaker_ids, dtype=float)

    if init_params.get('balance_by_domain'):
        assert domain_ids is not None
        dom_weight = 1.0/np.bincount(domain_ids) 
        weights *= dom_weight[domain_ids]
        
    if init_params.get('balance_by_speaker'):
        spk_weight = 1.0/np.bincount(speaker_ids) 
        weights *= spk_weight[speaker_ids]

    return weights
