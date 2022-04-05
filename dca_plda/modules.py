import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
from sklearn import discriminant_analysis
from scipy.special import logit, expit
from scipy.sparse import coo_matrix, dia_matrix
from scipy import linalg

from dca_plda import utils 
from dca_plda import data as ddata 
from dca_plda import calibration
from dca_plda import htplda
from dca_plda import generative

def compute_loss(llrs, metadata=None, ptar=0.01, mask=None, loss_type='cross_entropy', return_info=False, 
    enrollment_ids=None, ids_to_idxs=None, class_name='class_id'):

    if metadata is None:
        assert mask is not None
        # The mask is assumed to have been generated using the Key class, 
        # so targets are labeled with 1, impostors with -1 and non-scored trials with 0.
        valid = mask != 0
        same_class = mask == 1
    else:
        if enrollment_ids is None:
            # The llrs are assumed to correspond to all vs all trials
            same_class, valid = utils.create_scoring_masks(metadata, class_name)
        else:
            # The llrs are assumed to correspond to all samples. The class ids in this case are
            # indices that have to be mapped to enrollment indexes. 
            class_ids = metadata[class_name].type(torch.int)
            same_class = utils.onehot_for_class_ids(class_ids, enrollment_ids, ids_to_idxs).T
            # All samples are valid. 
            valid = torch.ones_like(llrs).type(torch.bool)

    # Select the valid llrs and shift them to convert them to logits
    llrs   = llrs[valid]
    logits = llrs + logit(ptar)
    labels = same_class[valid]
    labels = labels.type(logits.type())

    ptart = torch.as_tensor(ptar)
    tarw = (labels == 1).float()
    impw = (labels == 0).float()
    
    if "weighted" in loss_type:
        # Weights determined so that positive and negative samples are also balanced within each detector
        tar_count_per_detector = torch.sum(same_class, axis=1, keepdim=True).float()
        imp_count_per_detector = same_class.shape[1] - tar_count_per_detector
        tarw *= (1/tar_count_per_detector * torch.ones_like(same_class))[valid]
        impw *= (1/imp_count_per_detector * torch.ones_like(same_class))[valid]
    
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
        self.config = config
        self.in_dim = in_dim

        front_dim = self.config.get('front_dim')
        if front_dim:
            # Affine layer at the front (used to fine tune the 6th layer in the embedding extractor)
            self.front_stage = Affine(in_dim, front_dim)
            in_dim = front_dim

        lda_dim = self.config.get('lda_dim')
        if lda_dim:
            # LDA stage enabled by default.
            self.lda_stage  = Affine(in_dim, lda_dim, self.config.get('activation_after_lda', 'l2norm'))
        else:
            lda_dim = in_dim

        plda_type = self.config.get('plda_type', 'quadratic_form') 

        if plda_type == 'quadratic_form':
            # In this case, the PLDA stage is given by the quadratic form (or, rather, second order
            # polynomial) that corresponds to the score computation for Gaussian PLDA.
            # Note that, in this case, the parameters are those of the quadratic form, rather than
            # those of the original Gaussian PLDA model (the between and within class covariance 
            # matrices). Even though we initialize the parameters of the quadratic form using the
            # covariance matrices estimated with the EM algorithm, after discriminative training the 
            # parameters may no longer correspond to valid covariance matrices. Hence, we loose the 
            # probabilistic interpretation of the PLDA model.  
            self.plda_stage = Quadratic(lda_dim)

        elif plda_type == 'quadratic_form_non_symmetric_L':
            # Allow L to be assymetric after discriminative training
            self.plda_stage = Quadratic(lda_dim, non_symmetric_L=True)
        
        elif plda_type == 'heavy_tailed_plda':
            # In this case, the PLDA stage is given by the HT-PLDA formulation and uses the parameters
            # of the generative model: F and Cw. Hence, after discriminative training, we can still
            # use the resulting model to compute all sorts of probabilities. 
            # Note that, when nu tends to infinity, this model degenerates to Gaussian PLDA, but it 
            # will not coincide with the model above because the parameterization is different.
            self.plda_stage = HTPLDA(lda_dim, rank=self.config.get('htplda_rank', lda_dim))
        
        else:   
            raise Exception("Plda_type %s no implemented"%plda_type)

        if self.config.get('fixed_enrollment_ids'):
            # In this case, the enrollment vectors are pre-defined and another parameter of the model
            # Use the Affine class for this. Only the W of and the init methods of that class are used, not the forward.
            # Read the list of enrollment classes from a text file provided in the config.
            if not type(self.config.get('fixed_enrollment_ids')) in [tuple, list]:                 
                print("Reading fixed enrollment ids from %s"%self.config.get('fixed_enrollment_ids'))
                self.enrollment_classes = [l.strip().split()[0] for l in open(self.config.get('fixed_enrollment_ids')).readlines()]
                self.config['fixed_enrollment_ids'] = self.enrollment_classes
            else:
                self.enrollment_classes = self.config.get('fixed_enrollment_ids')
            #self.enrollment = Affine(lda_dim, len(self.enrollment_classes), bias=False)
            if self.config.get("enrollment_before_lda"):
                self.enrollment = Stats(in_dim, len(self.enrollment_classes))
                self.enrollment_before_lda = True
            else:
                self.enrollment = Stats(lda_dim, len(self.enrollment_classes))
                self.enrollment_before_lda = False
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

    def score(self, xt, xe=None, durt=None, dure=None, si_only=False, raw=False, subset_enrollment_idxs=None):
        # Same as the forward method but it allows for assymetric scoring where the rows and columns
        # of the resulting score file corresponds to two different sets of data

        hase = xe is not None

        if hasattr(self, 'front_stage'):
            xe = self.front_stage(xe) if hase else None
            xt = self.front_stage(xt)

        if hasattr(self, 'lda_stage'):
            x2e = self.lda_stage(xe) if hase else None
            x2t = self.lda_stage(xt)
        else:
            x2e = xe
            x2t = xt

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
                if subset_enrollment_idxs is not None:
                    enrollment_M = self.enrollment.M.T[subset_enrollment_idxs]
                else:
                    enrollment_M = self.enrollment.M.T
                if self.enrollment_before_lda:
                    enrollment_M = self.lda_stage(enrollment_M)
                scrs = self.plda_stage.score_with_stats(x2t, enrollment_M, self.enrollment.N)
                # In this case, there is no dur or si for the enrollment side. 
                # Set them to a vector of constants 
                dure = torch.ones(scrs.shape[0]).to(scrs.device) if durt is not None else None
                sie  = torch.ones([scrs.shape[0], sit.shape[1]]).to(scrs.device) if sit is not None else None
            else:
                scrs = self.plda_stage.score(x2t, x2e)

            llrs = self.cal_stage.calibrate(scrs, durt, sit, dure, sie)            
            return llrs if raw is False else scrs


    def init_params_with_data(self, embeddings, metadata, metamaps, trn_config, device=None, label_name='class_id', sample_weights=None, new_enrollment_classes=None):

        balance_method = trn_config.get("balance_method_for_batches")
        assert 'init_params' in trn_config
        init_params = trn_config.init_params
        init_components = init_params.get('init_components', 'all')

        if init_components == 'all':
            init_components = ['front', 'lda', 'si', 'plda', 'enrollment', 'calibration']

        # The code here repeats the steps in forward above, but adds the steps necessary for initialization.
        # I chose to keep these two methods separate to leave the forward small and easy to read.
        with torch.no_grad():

            x = utils.np_to_torch(embeddings, device)

            class_ids  = metadata[label_name]
            domain_ids = metadata['domain_id']
            
            if hasattr(self,'front_stage'):
                if 'front' in init_components:
                    print("Initializing front stage")
                    params_to_init_front_stage = trn_config.get("params_to_init_front_stage")
                    if params_to_init_front_stage:
                        # Read the params from an npz file
                        self.front_stage.init_with_params(np.load(params_to_init_front_stage))
                    else:
                        self.front_stage.init_random(init_params.get('stdev',0.1))
                x = self.front_stage(x)
                
            if hasattr(self,'lda_stage'):
                if 'lda' in init_components:
                    print("Initializing LDA stage")
                    if init_params.get("random"):
                        self.lda_stage.init_random(init_params.get('stdev',0.1))
                    else:
                        self.lda_stage.init_with_lda(x.cpu().numpy(), class_ids, init_params, sec_ids=domain_ids, sample_weights=sample_weights)
                x2 = self.lda_stage(x)
            else:
                x2 = x

            if hasattr(self,'si_stage1'):
                if self.si_input == 'main_input':
                    si_input = x
                else:
                    si_input = x2

                if 'si' in init_components:
                    print("Initializing side-info stages")
                    if init_params.get("random"):
                        self.si_stage1.init_random(init_params.get('stdev',0.1))
                    else:
                        self.si_stage1.init_with_lda(si_input.cpu().numpy(), class_ids, init_params, sec_ids=domain_ids, complement=True, sample_weights=sample_weights)

                s2 = self.si_stage1(si_input)
                    
                if 'si' in init_components:
                    if init_params.get('init_si_stage2_with_domain_gb', False):
                        # Initialize the second stage of the si-extractor to be a gaussian backend that predicts
                        # the posterior of each domain. In this case, the number of domains has to coincide with the
                        # dimension of the side info vector
                        assert self.si_dim == len(np.unique(domain_ids))
                        self.si_stage2.init_with_lda(s2.cpu().numpy(), domain_ids, init_params, sec_ids=class_ids, gaussian_backend=True, sample_weights=sample_weights)
        
                    else:
                        # This is the only component that is initialized randomly unless otherwise indicated by the variable "init_si_stage2_with_domain_gb"
                        self.si_stage2.init_random(init_params.get('w_init', 0.5), init_params.get('b_init', 0.0), init_params.get('type', 'normal'))

                if hasattr(self,'shift_selector'):
                    # Initialize the shifts as the mean of the lda outputs weighted by the si
                    si = self.si_stage2(s2)
                    if 'si' in init_components:
                        if init_params.get("random"):
                            self.shift_selector.init_random(init_params.get('stdev',0.1))
                        else:
                            self.shift_selector.init_with_weighted_means(x2.cpu().numpy(), si.cpu().numpy())
                    x2 -= self.shift_selector(si)

            if 'plda' in init_components:
                print("Initializing PLDA stage")
                if init_params.get("random"):
                    self.plda_stage.init_random(init_params.get('stdev',0.1))
                else:    
                    self.plda_stage.init_with_plda_trained_generatively(x2.cpu().numpy(), class_ids, init_params, domain_ids=domain_ids, sample_weights=sample_weights)

            if self.enrollment is not None and 'enrollment' in init_components:
                print("Initializing enrollment stats")
                if init_params.get("random"):
                    self.enrollment.init_random(init_params.get('stdev',0.1))
                else:    
                    if self.enrollment_before_lda:
                        enrollment_data = x.cpu().numpy()
                    else:
                        enrollment_data = x2.cpu().numpy()

                    replace = False
                    if init_params.get("enroll_new_classes"):
                        # Used to add new classes to a preexisting model
                        replace = True
                        self.enrollment_classes = list(metamaps[label_name+'_inv'].keys())
                        if new_enrollment_classes:
                            # If this list is provided, check that it is the same as the one obtained from the data
                            # except maybe in a different order
                            if set(new_enrollment_classes) != set(self.enrollment_classes):
                                raise Exception("List of new_enrollment_classes provided for init does not coincide with what is found in the data")                            
                            self.enrollment_classes = new_enrollment_classes
                        self.config['fixed_enrollment_ids'] = self.enrollment_classes
                         
                    self.enrollment.init_with_data(enrollment_data, class_ids, domain_ids, self.enrollment_classes, metamaps[label_name+'_inv'], init_params, replace=replace, sample_weights=sample_weights)

            # Since the training data is usually large, we cannot create all possible trials for x3.
            # So, to create a bunch of trials, we just create a trial loader with a large batch size.
            # This means we need to rerun the front stage and lda again, but it is a small price to pay for the 
            # convenience of reusing the machinery of trial creation in the TrialLoader.
            batch_size = trn_config.get("num_samples_for_calibration", 2000)
            if batch_size == 'all':
                # Batch size needs to be a multiple of num_samples_per_class
                batch_size = int(np.floor(len(class_ids)/trn_config.num_samples_per_class)*trn_config.num_samples_per_class)
            loader = ddata.TrialLoader(embeddings, metadata, metamaps, device, seed=0, batch_size=batch_size, num_batches=1, balance_method=balance_method, check_count_per_sess=False)
            x, meta_batch = next(loader.__iter__())
            
            x  = self.front_stage(x) if hasattr(self,'front_stage') else x
            x2 = self.lda_stage(x)   if hasattr(self,'lda_stage') else x
            if self.enrollment_classes:
                enrollment_M = self.enrollment.M.T
                if self.enrollment_before_lda:
                    enrollment_M = self.lda_stage(enrollment_M)
                scrs = self.plda_stage.score_with_stats(x2, enrollment_M, self.enrollment.N)
                class_ids = meta_batch[label_name].type(torch.int)
                same_class = utils.onehot_for_class_ids(class_ids, self.enrollment_classes, metamaps[label_name+'_inv']).T.type(torch.bool)
                valid = torch.ones_like(scrs).type(torch.bool)
            else:
                scrs = self.plda_stage.score(x2)
                # The llrs are assumed to correspond to all vs all trials
                same_class, valid = utils.create_scoring_masks(meta_batch)

            if 'calibration' in init_components:
                print("Initializing calibration stage")
                if init_params.get("random"):
                    self.cal_stage.init_random(init_params.get('stdev',0.1))
                else:
                    scrs_np, same_class_np, valid_np = [v.detach().cpu().numpy() for v in [scrs, same_class, valid]]
                    self.cal_stage.init_with_logreg(scrs_np, same_class_np, valid_np, trn_config.ptar, std_for_mats=init_params.get('std_for_cal_matrices',0))

            dummy_durs_t = torch.ones(scrs.shape[1]).to(device) 
            dummy_durs_e = torch.ones(scrs.shape[0]).to(device) 
            dummy_si_t = torch.zeros(scrs.shape[1], self.si_dim).to(device)
            dummy_si_e = torch.zeros(scrs.shape[0], self.si_dim).to(device)
            llrs = self.cal_stage.calibrate(scrs, dummy_durs_t, dummy_si_t, dummy_durs_e, dummy_si_e)
            mask = torch.ones_like(same_class, dtype=torch.int8)
            mask[~same_class] = -1
            mask[~valid] = 0

            return compute_loss(llrs, mask=mask, ptar=trn_config.ptar, loss_type=trn_config.loss)



class Hierarchical_DCA_PLDA_Backend(nn.Module):
    def __init__(self, in_dim, config):
        """
        Implements a hierachical version of DCA-PLDA where there's a first
        detector for clusters of classes, then there's a separate mean norm 
        for each cluster and a common DCA-PLDA after that to get the final LLR
        for each class as a function of the LLR from the first and second levels.
        See the score method for details on how the two LLRs are combined.
        """

        super().__init__()
        self.config = config
        self.in_dim = in_dim

        # For hierachical DCA-PLDA, the fixed_enrollment_ids file should have
        # one line for each class, with two columns: class_id cluster_id
        try:
            self.class_to_cluster_map = utils.get_class_to_cluster_map_from_config(config)
            config["class_to_cluster_map"] = self.class_to_cluster_map
        except:
            raise Exception("For hierarchical DCA-PLDA, file fixed_enrollment_ids (%s) in the config should have two columns, one with the class_id and a second one with the cluster_id for that class"%config.get("fixed_enrollment_ids"))

        config_level2 = utils.AttrDict(config.copy())
        config_level1 = utils.AttrDict(config.copy())
        config_level1['fixed_enrollment_ids'] = list(np.sort(np.unique(list(self.class_to_cluster_map.values()))))
        self.level1_detector = DCA_PLDA_Backend(in_dim, config_level1)
        
        # The second level is modeled as a cluster-dependent mean norm, followed by
        # a common DCA_PLDA backend
        self.cluster_stats = Stats(in_dim, len(self.level1_detector.enrollment_classes))

        self.enrollment_classes = []
        self.logRcs  = []
        self.logRls  = []
        self.logRlcs = []
        self.level2_enrollment_idxs = []
        last_idx = 0
        for i, cluster in enumerate(self.level1_detector.enrollment_classes):
            classes = list(np.sort([l for l,c in self.class_to_cluster_map.items() if c==cluster]))
            self.enrollment_classes += classes
            if config.get("uniform_priors_by_language"):
                # The prob of each cluster is proportional to the number of languages it has
                logRc = torch.tensor([logit(len(classes)/len(self.class_to_cluster_map))])
            else:
                # The prob of each cluster is the same for all clusters
                logRc = torch.tensor([logit(1.0/len(self.level1_detector.enrollment_classes))])
            # The languages within each cluster are always equally likely
            logRlc = torch.tensor([logit(1.0/len(classes))])
            logRl  = self._output_logodds_from_inner_logodds(logRc, logRlc, check_inf=True) 
            self.logRlcs.append(logRlc)
            self.logRls.append(logRl)
            self.logRcs.append(logRc)
            self.level2_enrollment_idxs.append(np.arange(last_idx,last_idx+len(classes)))
            last_idx += len(classes)
            print("Priors for cluster %s: cluster = %4.2f, language_cond_to_cluster = %4.2f, language = %4.2f"%(cluster, 
                expit(logRc.detach().cpu().numpy()), expit(logRlc.detach().cpu().numpy()), expit(logRl.detach().cpu().numpy())))

        self.logRcs = torch.unsqueeze(torch.tensor(self.logRcs),1)
        config_level2['fixed_enrollment_ids'] = self.enrollment_classes
        # By default, the lda_dim of the second level is the same as for the first level
        config_level2['lda_dim'] = config.get("lda_dim_level2", config.get("lda_dim"))
        self.level2_detector = DCA_PLDA_Backend(in_dim, config_level2)
        
    def forward(self, x, durations=None, all_levels=False):
        # The forward method assumes the same enroll and test data. This is used in training.
        return self.score(x, durt=durations, all_levels=all_levels)

    @staticmethod
    def _output_logodds_from_inner_logodds(lo1, lo2, check_inf=False):
        # Computes log(o1 o2 / (o1+o2+1))
        # from lo1 = log(o1) and lo2 = log(o2)
        # If either loi is inf (ie, oi is inf=1/0), then the output should be the other lo    

        assert lo1.shape[0] == 1

        lo = lo1 + lo2 - torch.logaddexp(torch.logaddexp(lo1, lo2), torch.zeros(1))

        if check_inf:
            lo1_ispi = lo1.isposinf()
            lo[:,lo1_ispi[0]] = lo2[:,lo1_ispi[0]] 

            lo2_ispi = lo2.isposinf()
            if lo2.shape[0]>1:
                lo[lo2_ispi] = lo1[:,lo2_ispi[0]] 
            else:
                lo[lo2_ispi] = lo1[lo2_ispi]

        return lo


    def score(self, xt, xe=None, durt=None, dure=None, raw=None, all_levels=False, cluster_prior_dict=None):
        """ The posterior odds for class l is given by:
        ol = oc olc / (oc + olc + 1)
        where l is the class, c is the cluster, x are the features,
        oc are the odds for cluster c, which are obtained from the
        first level detector, and olc are the odds for class l given 
        cluster c, which are obtained from the second level detector. 

        Hierarchical PLDA can only be used with fixed enrollment ids, so
        xe and dure should never be provided. We leave them as arguments so 
        that the wrapper scripts for standard DCA-PLDA can be used without change.
        """
    
        assert xe is None and dure is None
        llrc = self.level1_detector.score(xt, durt=durt)
        
        if cluster_prior_dict is not None:
            logRcs = torch.tensor([logit(cluster_prior_dict[c]) for c in self.level1_detector.enrollment_classes])
            print("Using the following log prior odds for clusters: %s"%" ".join(["%s:%f"%(c, logRcs[i]) for i, c in enumerate(self.level1_detector.enrollment_classes)]))
            # Given that logRc has changed, we also need to recompute the logRls instead
            # of using the ones defined in the init method
            logRls = []
            for i, cluster in enumerate(self.level1_detector.enrollment_classes):
                logRls.append(self._output_logodds_from_inner_logodds(logRcs[i:i+1], self.logRlcs[i], check_inf=True))
            # Add a last dimension so that we can sum it with llrc
            logRcs = torch.unsqueeze(logRcs,1)
        else:
            # If not provided, use the one computed at init
            logRcs = self.logRcs
            logRls = self.logRls
    
        # Log odds for each cluster
        loc = llrc + logRcs
        
        llrl_list  = []
        llrlc_list = []
        for i, cluster in enumerate(self.level1_detector.enrollment_classes):
            # First, center the features according to the mean for this cluster
            xt_norm = xt - self.cluster_stats.M[:,i]
            # Get the cluster-dependent LLR
            if self.logRlcs[i].isposinf():
                # This happens when there's a single language in the cluster
                # Slice it like this to keep the dimension
                # While this could be done transparently by the code in the else statement
                # we keep it separate to avoid computing llrlc, which is unnecesary since
                # lolc will be inf regardless of the value of llrlc.
                lol = loc[i:(i+1)]
                llrlc_list.append(torch.zeros_like(lol))
            else:
                llrlc = self.level2_detector.score(xt_norm, durt=durt, subset_enrollment_idxs=self.level2_enrollment_idxs[i])
                lolc  = llrlc + self.logRlcs[i]
                # Compute the cluster-independent LLR. Checking for inf is only needed when cond_to_clusters is set.
                lol   = self._output_logodds_from_inner_logodds(loc[i:(i+1)], lolc, check_inf=cluster_prior_dict is not None)
                llrlc_list.append(llrlc)

            llrl = lol - logRls[i]
            llrl_list.append(llrl)
        
        all_llrl = torch.cat(llrl_list)

        if all_levels:
            all_llrlc = torch.cat(llrlc_list)
            return all_llrl, llrc, all_llrlc
        else:
            return all_llrl


    def init_params_with_data(self, embeddings, metadata, metamaps, trn_config, device=None):

        init_params = trn_config.init_params
        init_components = init_params.get('init_components', 'all')

        with torch.no_grad():

            cluster_names = [metamaps['cluster_id'][c] for c in metadata['cluster_id']]
            class_names   = [metamaps['class_id'][c]   for c in metadata['class_id']]

            new_enrollment_classes = None
            if init_params.get("enroll_new_classes"):
                self.class_to_cluster_map = dict(np.unique(np.c_[class_names,cluster_names],axis=0)) 
                self.config["class_to_cluster_map"] = self.class_to_cluster_map
                # Exact same procedure as in __init__ so that when the model is loaded, classes are ordered the same way
                new_enrollment_classes = list(np.sort(np.unique(list(self.class_to_cluster_map.values()))))

            cla_ids = metadata['class_id']
            dom_ids = metadata['domain_id']    
            sample_weights = compute_weights(cla_ids, dom_ids, init_params.get("balance_method"))['sample_weights']    
            if init_params.get("balance_method") == 'same_num_samples_per_class_and_dom':
                # For level1, we want each class in each cluster to weight the same so that the mean of the per-language means is the
                # same as the (weighted) cluster mean. Since the per-language means are computed with equal weight per domain, we need
                # to do the same here.
                # First, get the weight as for the output layer classes (ie, each domain x class pair gets the same weight).
                # These weights give more importance to classes that appear in more domains. Hence, we then multiply those weights
                # by the inverse of the number of domains that each language appears in 
                for cla in np.unique(cla_ids):
                    dom_for_cla = np.unique(dom_ids[cla_ids==cla])
                    sample_weights[cla_ids==cla] *= 1/len(dom_for_cla)

                # For debugging
                # domain_names  = [metamaps['domain_id'][c]  for c in metadata['domain_id']]
                # print("Weights per class, cluster and domain")
                # utils.print_weights_by_class([s+','+c+','+d for s,c,d in zip(class_names, cluster_names, domain_names)], sample_weights)
                # print("Weights per class and cluster")
                # utils.print_weights_by_class([s+','+c for s,c in zip(class_names, cluster_names)], sample_weights)
                # print("Weights per cluster")
                # utils.print_weights_by_class(cluster_names, sample_weights)

            loss1 = self.level1_detector.init_params_with_data(embeddings, metadata, metamaps, trn_config, device, 'cluster_id', 
                        new_enrollment_classes=new_enrollment_classes, sample_weights=sample_weights)

            # Initialize the means of each cluster. 
            if init_components == 'all' or 'enrollment' in init_components:
                print("Initializing cluster means")
                replace = False
                new_enrollment_classes = None
                if init_params.get("enroll_new_classes"):
                    # Replace the enrollment stats with new ones
                    replace = True
                    # Make sure the level2 enrollment classes are sorted by cluster
                    self.enrollment_classes = []
                    self.level2_enrollment_idxs = []
                    last_idx = 0
                    for i, cluster in enumerate(self.level1_detector.enrollment_classes):
                        classes = list(np.sort([l for l,c in self.class_to_cluster_map.items() if c==cluster]))
                        self.enrollment_classes += classes
                        self.level2_enrollment_idxs.append(np.arange(last_idx,last_idx+len(classes)))
                        last_idx += len(classes)

                self.cluster_stats.init_with_data(embeddings, metadata['cluster_id'], metadata['domain_id'], self.level1_detector.enrollment_classes, 
                                    metamaps['cluster_id_inv'], init_params, replace=replace, sample_weights=sample_weights)

            # Now, get the mean corresponding to the true cluster of each sample
            cluster_indexes = [self.level1_detector.enrollment_classes.index(c) for c in cluster_names]
            cluster_means   = self.cluster_stats.M[:, cluster_indexes].detach().cpu().numpy().T

            loss2 = self.level2_detector.init_params_with_data(embeddings-cluster_means, metadata, metamaps, trn_config, device, 'class_id', new_enrollment_classes=self.enrollment_classes)
            
        level1_loss_weight = trn_config.get("level1_loss_weight", 0.0)
        return level1_loss_weight * loss1 + (1-level1_loss_weight) * loss2



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

            self.transform_type = config.get('transform_type', 'quadratic_form')
            self.non_symmetric_L = self.transform_type == 'quadratic_form_non_symmetric_L'

        if self.config and self.config.get('sideinfo_dependent'):
            self.is_sidep = True
            use_G = config.get("use_G_for_si", True) 
            use_L = config.get("use_L_for_si", True) 
            use_C = config.get("use_C_for_si", True)
            self.si_dim = si_dim
            if self.config.get('concat_durfeats_with_sideinfo'):
                self.si_dim += numdurf
            self.sidep_alpha = Quadratic(self.si_dim, use_G=use_G, use_C=use_C, use_L=use_L, non_symmetric_L=self.non_symmetric_L)
            self.sidep_beta  = Quadratic(self.si_dim, use_G=use_G, use_C=use_C, use_L=use_L, non_symmetric_L=self.non_symmetric_L)

        if self.config and self.config.get('duration_dependent'):
            self.is_durdep = True
            # Fake input to _process_durs just to get the dimension of the processed duration vector
            use_G = config.get("use_G_for_dur", True) 
            use_L = config.get("use_L_for_dur", True) 
            use_C = config.get("use_C_for_dur", True) 
            self.durdep_alpha = Quadratic(numdurf, use_G=use_G, use_C=use_C, use_L=use_L, non_symmetric_L=self.non_symmetric_L)
            self.durdep_beta  = Quadratic(numdurf, use_G=use_G, use_C=use_C, use_L=use_L, non_symmetric_L=self.non_symmetric_L)
        
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


    def init_with_logreg(self, scores, same_class, valid, ptar, std_for_mats=0):

        scores = scores[valid]
        labels = same_class[valid]

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
    def __init__(self, in_dim, use_G=True, use_C=True, use_L=True, non_symmetric_L=False):
        super().__init__()
        self.L = Parameter(torch.zeros(in_dim, in_dim)) if use_L else None
        self.G = Parameter(torch.zeros(in_dim, in_dim)) if use_G else None
        self.C = Parameter(torch.zeros(in_dim, 1)) if use_C else None
        self.k = Parameter(torch.tensor(0.0))
        self.non_symmetric_L = non_symmetric_L

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
            L = self.L if self.non_symmetric_L else 0.5*(self.L+self.L.T)
            Lterm = torch.matmul(torch.matmul(xe, L), xt.T)
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

    def score_with_stats(self, xt, M, N):
        # In this PLDA form we cannot do proper PLDA scoring because we loose the covariance matrices.
        # Hence, we simply represent the embeddings from each sample by their mean. The N is ignored.
        return self.score(xt, M)

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

    def init_with_plda_trained_generatively(self, x, class_ids, init_params, domain_ids=None, sample_weights=None):
        # Compute a PLDA model with the input data, approximating the 
        # model with just the usual initialization without the EM iterations

        weights = compute_weights(class_ids, domain_ids, init_params.get('balance_method'), sample_weights)

        # Use the cross-product between class and domain ids for initializing LDA and PLDA
        # This can be used to be able to increase the LDA dimension beyond the number of available
        # classes 
        if init_params.get('lda_by_class_x_domain'):
            # Replace the class_ids with new indices that correspond to the cross-product between
            # class_ids and sec_ids. In this case, the balance_method option does not make sense
            assert not init_params.get('balance_method') and sample_weights is None
            _, init_ids = np.unique([str(s)+','+str(c) for s,c in zip(class_ids, domain_ids)], return_inverse=True)
            # Need to recompute the weights (which will all be 1.0 since balance_method is false), since
            # we now have more classes
            weights = compute_weights(init_ids, domain_ids, init_params.get('balance_method'))
        else:
            init_ids = None

        # Bi and Wi are the between and within covariance matrices and mu is the global (weighted) mean
        Bi, Wi, mu = generative.compute_2cov_plda_model(x, class_ids, weights, init_params.get('plda_em_its',0), init_ids=init_ids)
        
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



class HTPLDA(nn.Module):

    def __init__(self, in_dim, rank):
        super().__init__()
        self.F = Parameter(torch.zeros(in_dim, rank)) 
        self.H = Parameter(torch.zeros(in_dim, in_dim))
        self.mu = Parameter(torch.zeros(in_dim,1))
        self.nu = Parameter(torch.tensor(1.0))
        self.rank = rank

    def forward(self, x):
        # The forward method assumes the same enroll and test data. This is used in training.
        return self.score(x)

    def score(self, xt, xe=None):
 
        if xe is None:
            xe = xt
        
        xem = xe-self.mu.T
        xtm = xt-self.mu.T

        return htplda.score_matrix(self.H, self.F, self.nu, xem.T, xtm.T)

    def score_with_stats(self, xt, M, N):

        Me  = M - self.mu.T
        Ne  = N
        xtm = xt - self.mu.T

        return htplda.score_matrix_with_stats(self.H, self.F, self.nu, Me.T, xtm.T, Ne)


    def init_with_plda_trained_generatively(self, x, class_ids, init_params, domain_ids=None):
        # Compute a PLDA model with the input data, approximating the 
        # model with just the usual initialization without the EM iterations

        if init_params.get('balance_method'):
            raise Exception("Balance methos not implemented for HT-PLDA initialization")

        class_weights = None
        nu = init_params.get('htplda_nu', 10000)

        mu, F, Cw = generative.compute_ht_plda_model(x, class_ids, self.rank, nu, class_weights, niters=5, quiet=False)

        L = np.linalg.cholesky(Cw)
        I = np.identity(Cw.shape[0])
        H = linalg.solve_triangular(L, I, trans='T', lower=True)
            
        state_dict = {'F': F, 'H': H, 'mu': mu[:,np.newaxis], 'nu': nu}

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

    def init_with_lda(self, x, class_ids, init_params, complement=True, sec_ids=None, gaussian_backend=False, sample_weights=None):

        if self.has_bias is False:
            raise Exception("Cannot initialize this component with init_with_lda because it was created with bias=False")

        if init_params.get('lda_by_class_x_domain'):
            # Replace the class_ids with new indices that correspond to the cross-product between
            # class_ids and sec_ids. In this case, the balance_method option does not make sense
            assert not init_params.get('balance_method') is not True
            _, class_ids = np.unique([str(s)+','+str(c) for s,c in zip(class_ids, sec_ids)], return_inverse=True)

        weights = compute_weights(class_ids, sec_ids, init_params.get('balance_method'), sample_weights)

        BCov, WCov, GCov, mu, mu_per_class, _ = generative.compute_lda_model(x, class_ids, weights)

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


class Stats(nn.Module):
    def __init__(self, in_dim, num_classes):
        # Store 1st and 0th order stats from data
        super().__init__()
        self.M = Parameter(torch.zeros(in_dim, num_classes))
        self.N = Parameter(torch.zeros(num_classes))
      
    def init_with_data(self, data, class_ids, domain_ids, enrollment_classes, class_id_map, init_params, replace=False, sample_weights=None):

        # class_id_map contains a mapping from string to indices corresponding to the class_ids array
        weights = compute_weights(class_ids, domain_ids, init_params.get('balance_method'), sample_weights)['sample_weights'][:,np.newaxis]
        class_ids_onehot = utils.onehot_for_class_ids(class_ids, enrollment_classes, class_id_map)
        weights = class_ids_onehot * weights
        counts = np.sum(weights,axis=0)
        means = data.T @ weights / counts

        if init_params.get('fix_enrollment_counts_to_one'):
            # Used for comparing the effect of taking into account the counts
            counts = np.ones_like(counts)

        if replace:
            # Need to recreate M and N with the right dimensions
            num_classes = len(counts)
            in_dim = means.shape[0]
            self.M = Parameter(torch.zeros(in_dim, num_classes))
            self.N = Parameter(torch.zeros(num_classes))

        utils.replace_state_dict(self, {'M': means, 'N': counts})

    @staticmethod
    def create_with_values(M, N):
        in_dim, num_classes = M.shape
        stats = Stats(in_dim, num_classes)
        utils.replace_state_dict(stats, {'M': M, 'N': N})
        return stats

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



def compute_weights(class_ids, domain_ids, balance_method=False, external_sample_weights=None):
    
    class_weights = None
    sample_weights = None
    unique_cids = np.unique(class_ids)
    unique_doms = np.unique(domain_ids)

    assert np.max(class_ids)+1 == len(unique_cids)
    assert np.max(domain_ids)+1 == len(unique_doms)

    print("Computing class and sample weights with balance_method = %s"%balance_method)

    if balance_method not in ['none', 'same_num_samples_per_class_and_dom', 'same_num_samples_per_class_then_same_num_samples_per_dom', 'same_num_classes_per_dom_then_same_num_samples_per_class']:
        raise Exception("Balance method %s not implemented"%balance_method)

    if external_sample_weights is not None:
        sample_weights = external_sample_weights.copy()
    else:
        sample_weights = np.ones_like(class_ids, dtype=float)
        class_x_dom = [str(s)+','+str(c) for s,c in zip(class_ids, domain_ids)]
        if balance_method == 'same_num_samples_per_class_and_dom':
            # Assign one weight per sample, so that all classes in each domain and
            # accross domains weight the same. That is, each sample is weighted by
            # the inverse of the number of samples for that class and domain.
            # If all classes appear in a single domain, this is very similar to
            # the per-class implementation of the weights below.
            # This can also be used to balance out the language-id samples so that all 
            # languages have the same equivalent number of samples for each domain they 
            # appear in.
            sample_weights *= utils.compute_weights_to_balance_by_class(class_x_dom)
        elif balance_method == 'same_num_samples_per_class_then_same_num_samples_per_dom':
            # Assign one weight per sample so that all classes weight the same
            # and, within each class, each domain weights the same
            # We simply compute a weight that is the inverse of the count for each 
            # class/domain pair (as above) and then further divide it by the number 
            # of domains for each class.
            sample_weights *= utils.compute_weights_to_balance_by_class(class_x_dom)
            num_dom_per_class =  np.array([len(np.unique(domain_ids[class_ids==c])) for c in unique_cids])
            sample_weights *= 1/num_dom_per_class[class_ids]
        #utils.print_weights_by_class(class_x_dom, sample_weights)

    dom_weight = np.ones_like(unique_doms, dtype=float)
    for d in unique_doms:
        if balance_method is True or balance_method == 'same_num_classes_per_dom_then_same_num_samples_per_class':
            # The weight for each domain is given by the inverse of the number of classes for that domain    
            # If all domains have the same number of classes, then all weights are 1.0
            dom_weight[d] = len(unique_cids)*1.0/len(np.unique(class_ids[domain_ids==d]))/len(unique_doms)
        else:
            # Dummy weight of 1 for all domains
            dom_weight[d] = 1.0

    # The weight for each class is given by the weight for the domain to which it belongs.
    # This assumes each class belongs to a single domain, which will mostly be the case for speaker id,
    # though not for language id. 
    class_weights = np.ones_like(unique_cids, dtype=float)
    for s in unique_cids:
        class_weights[s] = dom_weight[domain_ids[np.where(class_ids==s)[0][0]]]

    print("Weights per domain: %s"%dom_weight)

    return {'class_weights': class_weights, 'sample_weights': sample_weights}

