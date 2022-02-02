"""
Code based on the sitw scorer code by Mitch McLaren, with some additions and changes by Luciana Ferrer
"""

import sys, os, re, gzip, h5py
import numpy as np
import gzip
from scipy.optimize import minimize
from scipy.special import logit
from dca_plda import calibration, utils
import random

class Key(object):
    """
    A Key object to filter tar/non scores from a scorefile
    The object is in full-matrix form ie: it contains
    enroll_ids: list of string of modelids
    test_ids: list of string of test_ids
    mask: a len(enroll_ids)xlen(test_ids) int matrix containing
        +1: for target trial
        -1: for impostor trial
        0: for non scored trial
    """
    def __init__(self, enrollids, testids, mask):
        self.enroll_ids=list(enrollids)
        self.test_ids=list(testids)
        if(mask.shape!=(len(enrollids),len(testids))):
            raise Exception("Incompatible mask for creation of key")
        self.mask=mask.astype(np.int8)

    @classmethod
    def load(self, filename, enrollids=None, testids=None):
        """ Build Key from a text file with the following format 
        * trainID testID tgt/imp for SID
        * testID languageID for LID
        or in h5 format where the enrollids are the languageids in the case of LID.
        """

        if filename.endswith(".h5"):
            with h5py.File(filename, 'r') as f:
                enrollids_in_key = f['enroll_ids'][()] if 'enroll_ids' in f else f['train_ids'][()]
                testids_in_key   = f['test_ids'][()]
                mask_in_key      = f['trial_mask'][()]

                if type(enrollids_in_key[0]) == np.bytes_:
                    enrollids_in_key = [i.decode('UTF-8') for i in enrollids_in_key]
                    testids_in_key   = [i.decode('UTF-8') for i in testids_in_key]
        else:
            func = gzip.open if os.path.splitext(filename)[1] == ".gz" else open
            with func(filename) as f:
                lines = [line.strip().split(' ')[0:3] for line in f]

            if len(lines[0]) == 3:
                enrollids_in_key, testids_in_key, _ = map(lambda t: list(set(t)), zip(*lines))
            elif len(lines[0]) == 2:
                testids_in_key, _ = map(lambda t: list(set(t)), zip(*lines))
                enrollids_in_key = enrollids
            else:
                raise Exception("Need 3 columns for a SID key or 2 for LID key (file %s)."%filename)
            mask_in_key = None

        print("Loaded key file %s"%filename)
        check_align = False
        if enrollids is not None:
            missing = set.difference(set(enrollids),set(enrollids_in_key))
            if len(missing)>0:
                print("  There were %d enrollment ids in the key missing from the input enrollids list. Deleted them from key."%len(missing))
            check_align = True
        else:
            enrollids = enrollids_in_key

        if testids is not None:
            missing = set.difference(set(testids),set(testids_in_key))
            if len(missing)>0:
                print("  There were %d test ids in the key missing from the input testids list. Deleted them from key."%len(missing))
            check_align = True
        else:
            testids = testids_in_key

        if mask_in_key is not None:
            # If necessary, realign the key to the input test and enroll ids
            if check_align and (np.any(testids != testids_in_key) or np.any(enrollids != enrollids_in_key)):
                mask, _, _, _ = create_aligned_mask(mask_in_key, enrollids, testids, enrollids_in_key, testids_in_key)
            else:
                mask = mask_in_key

        else:
            idxenrollids = dict([(x,i) for i,x in enumerate(enrollids)])
            idxtestids = dict([(x,i) for i,x in enumerate(testids)])
            nbT, nbt = len(enrollids), len(testids)

            if len(lines[0]) == 3:
                # default mask to have all trials not included ('0')
                mask = np.zeros((nbT, nbt),dtype=np.int8)
                for e, t, vs in lines:
                    if e in idxenrollids and t in idxtestids:
                        v = 1 if vs == "tgt" else -1
                        mask[idxenrollids[e], idxtestids[t]] = v
            else:
                # default mask is 0, for those test files that do not
                # appear in the key, it stays 0
                mask = np.zeros((nbT, nbt),dtype=np.int8)
                for t, e in lines:
                    if e in idxenrollids and t in idxtestids:
                        # for the test files that do appear in the key, set
                        # the mask to -1 except for the true language
                        mask[:, idxtestids[t]] = -1
                        mask[idxenrollids[e], idxtestids[t]] = 1

        return Key(enrollids, testids, mask)


class Scores(object):
    """
    A Scores object to hold trial scores from a scorefile
    The object is in full-matrix form ie: it contains
    enroll_ids: list of string of modelids
    test_ids: list of string of test_ids
    score_mat: a len(enroll_ids)xlen(test_ids) float matrix containing all scores
    """
    def __init__(self, enrollids, testids, score_mat, missing=None):
        self.enroll_ids = list(enrollids)
        self.test_ids = list(testids)
        ntrain = len(enrollids)
        ntest  = len(testids)
        if score_mat.shape!=(ntrain,ntest):
            raise Exception("Incompatible score_mat size for creation of Scores")
        self.score_mat = score_mat
        if missing is not None:
            self.missing = missing
        else:
            self.missing = np.zeros([ntrain, ntest], dtype=int)
        

    def align(self, key):
        """
        Will align a Scores object with a key
        This will create a new scorefile, aligned with the keyfile
        Missing scores will be imputed with 0.0, which is a reasonable value
        if scores are calibrated.
        @param key: loaded Key object
        """

        new_score_mat, hase, hast, missing = create_aligned_mask(self.score_mat, key.enroll_ids, key.test_ids, self.enroll_ids, self.test_ids)
        nb_enroll = len(key.enroll_ids)
        nb_tests  = len(key.test_ids)

        if(np.sum(hase) < nb_enroll):
            print("WARNING: Missing %d models from your score file. Found %d but expected %d. Filled in those trials with 0.0." \
                % (nb_enroll-np.sum(hase),np.sum(hase),nb_enroll))

        if(np.sum(hast) < nb_tests):
            print("WARNING: Missing %d test segments from your score file. Found %d but expected %d. Filled in those trials with 0.0." \
                % (nb_tests-np.sum(hast),np.sum(hast),nb_tests))
        
        return Scores(key.enroll_ids, key.test_ids, new_score_mat, missing)

    def save(self, outfile, fmt = 'h5'):
        if fmt == 'h5':
            with h5py.File(outfile,'w') as f:
                f.create_dataset('test_ids',   data=np.string_(self.test_ids))
                f.create_dataset('enroll_ids', data=np.string_(self.enroll_ids))
                f.create_dataset('scores',     data=self.score_mat)
        else:
            with open(outfile, 'w') as f:
                for ti, t in enumerate(self.test_ids):
                    for ei, e in enumerate(self.enroll_ids):
                        print("%s %s %f"%(e,t, self.score_mat[ei, ti]), file=f)


    @classmethod
    def load(self, filename):
        """ Load scores from h5 file created by the save method.
        """

        with h5py.File(filename, 'r') as f:
            enrollids = f['enroll_ids'][()] if 'enroll_ids' in f else f['train_ids'][()]
            testids   = f['test_ids'][()]
            scores    = f['scores'][()]

            if type(enrollids[0]) == np.bytes_:
                enrollids = [i.decode('UTF-8') for i in enrollids]
                testids   = [i.decode('UTF-8') for i in testids]

        print("Loaded score file %s"%filename)

        return Scores(enrollids, testids, scores)


class IdMap(object):
    """
    Mapping between the enrollment/test model ids to the sample ids in the embeddings file.
    The mapping is saved as a dictionary of matrices. Each entry in the dict corresponds to one K, 
    where K is the number of sample ids that correspond to one enrollment/test model id. The matrix 
    for K is of size KxN where N is the numnber of unique enrollment/test ids and contains the sample
    indices needed to create that enrollment or test model. 
    """

    def __init__(self, sample_ids):
        self.sample_id_to_idx = dict(np.c_[sample_ids, np.arange(len(sample_ids))])
        self.sample_ids = sample_ids
        self.mappings = dict()
        self.model_ids = []
        self.model_dict = dict()

    @classmethod
    def load(self, mapfile, sample_ids):

        idmap = IdMap(sample_ids)

        missing = dict()

        if mapfile != "NONE":
            lines = [l.strip().split()[0:2] for l in open(mapfile).readlines()]
        else:
            # If no mapfile is provided, asume that the mapping includes all samples
            # and that model ids coincide with sample ids.
            lines = [(s,s) for s in sample_ids]

        for mid, sid in lines:
            if sid in idmap.sample_id_to_idx:
                if mid not in idmap.model_dict:
                    idmap.model_dict[mid] = [sid]
                else:
                    idmap.model_dict[mid].append(sid)
            else:
                missing[sid] = missing[sid] + 1 if sid in missing else 1

        mappings = []
        for K in np.unique([len(l) for l in idmap.model_dict.values()]):
            mids = []
            mmap = []
            for mid, sid_list in idmap.model_dict.items():
                if len(sid_list)==K:
                    mids.append(mid)
                    mmap.append([idmap.sample_id_to_idx[sid] for sid in idmap.model_dict[mid]])

            mmap = np.array(mmap, dtype=int).T
            print("Loaded %d %d-session models from %s"%(mmap.shape[1], K, mapfile))
            idmap.mappings[K] = {'model_ids': mids, 'map': mmap}
            idmap.model_ids += mids

        for sid, count in missing.items():
            print("  WARNING: id %s, which appears %d times in the loaded map file, is missing from the sample id list."%(sid, count))

        return idmap


def compute_performance(scores, keylist, outfile, ptar=0.01, setname=None, enrollment_ids=None):
    """
    Compute several performance metrics and print them in outfile
    """

    outf = open(outfile, "w")
    ptars = [ptar, 0.5]
 
    outf.write("%-32s                                            |         Ptar=%4.2f           |         Ptar=%4.2f           |    Ptar=%4.2f    |   Ptar=%4.2f  \n"%(" ",ptars[0],ptars[1], ptars[0], ptars[1]))
    outf.write("%-32s | #TGT(#missing) #IMP(#missing) |    EER   |   ACLLR MCLLR_LIN MCLLR_PAV |   ACLLR MCLLR_LIN MCLLR_PAV |  ADCF    MDCF   |   ADCF   MDCF  \n"%"Key")

    for keyf in [l.strip() for l in open(keylist).readlines()]:

        key  = Key.load(keyf, enrollids=enrollment_ids)
        name = re.sub('.h5$', '', os.path.basename(keyf))
        if setname is not None:
            name="%s:%s"%(setname,name)

        ascores = scores.align(key)
        tar = ascores.score_mat[key.mask==1]
        non = ascores.score_mat[key.mask==-1]
        det = Det(tar, non, pav=True)

        missing_tar = np.sum((key.mask==1)*ascores.missing)
        missing_non = np.sum((key.mask==-1)*ascores.missing)
        min_dcfs = det.min_dcf(ptars)
        act_dcfs = det.act_dcf(ptars)
        act_cllrs = det.act_cllr(ptars)
        min_cllrs = det.min_cllr(ptars)
        min_cllrs_pav = det.min_cllr(ptars, with_pav=True)
        #min_cllrs_pav = [np.nan, np.nan]
        eer = det.eer(from_rocch=True)
        outf.write("%-32s | %14s %14s |  %6.2f  |  %7.4f  %7.4f  %7.4f  |  %7.4f  %7.4f  %7.4f  | %7.4f %7.4f | %7.4f %7.4f \n"%(name, 
            "%d(%d)"%(len(det.tar), missing_tar), 
            "%d(%d)"%(len(det.non), missing_non), 
            eer*100, 
            act_cllrs[0], min_cllrs[0], min_cllrs_pav[0], 
            act_cllrs[1], min_cllrs[1], min_cllrs_pav[1], 
            act_dcfs[0], min_dcfs[0],
            act_dcfs[1], min_dcfs[1]))


def compute_performance_with_confidence_intervals(scores, keylist, outfile, logid2spk_dict, logid2ses_dict=None, ptar=0.01, setname=None, num_boot_samples=1000, percentile=5):
    """
    Compute EER, Actual and Min Cllr, and Actual and Min DCF (for the given ptar)
    including confidence intervals obtained with bootstrapping. The function needs
    a dictionary from logid to speaker id and another dictionary from logid to session
    id in order to generate samples by speaker and session as well as by logid.
    That is, each bootstrap sample is composed by a random selection (with replacement)
    of speakers, sessions and logids.
    If the logid2ses_dict is None, it does not include that part of the randomization.
    This should be used when there's a single logid per session in which case, randomizing
    the sessions and the logids would probably be wrong.
    This code is very slow for large datasets. It would have to be optimized (probably using
    indexes instead of strings) for use in those cases. 
    """

    outf = open(outfile, "w")
    outf.write("%-32s |    EER      EERp5    EERp95 |   ACLLR ACLLRp5 ACLLRp95 |   MCLLR   MCLLRp5  MCLLRp95  |   ADCF  ADCFp5  ADCFp95 |    MDCF  MDCFp5  MDCFp95 \n"%"Key")

    randomize_by_logid = True
    if logid2ses_dict is None:
        # If this dict is not provided, do not randomize both by session and logid, just do it once
        randomize_by_logid = False

    from IPython import embed


    for keyf in [l.strip() for l in open(keylist).readlines()]:
        key  = Key.load(keyf)
        name = re.sub('.h5$', '', os.path.basename(keyf))
        if setname is not None:
            name="%s:%s"%(setname,name)

        # Metrics for the full key
        ascores = scores.align(key)
        tar = ascores.score_mat[key.mask==1]
        non = ascores.score_mat[key.mask==-1]
        det_all = Det(tar, non, key)

        mdcf_all  = det_all.min_dcf(ptar)
        adcf_all  = det_all.act_dcf(ptar)
        mcllr_all = det_all.min_cllr(ptar)
        acllr_all = det_all.act_cllr(ptar)
        eer_all   = det_all.eer()*100

        # Keep only the speakers for which some logid is present in the key
        logid_list = np.unique(key.test_ids + key.enroll_ids)

        if not randomize_by_logid:
            # Assign a dummy session id to be the same as the logid
            logid2ses_dict = dict([(l, l) for l in logid_list])

        sesid_list = np.unique([logid2ses_dict[logid] for logid in logid_list]) 
        spkid_list = np.unique([logid2spk_dict[logid] for logid in logid_list])
        
        print("Original key, num_spk = %d, num_sess = %d, num_logids = %d, num_tgt = %d, num_imp = %d, acllr = %.2f"%
            (len(np.unique(spkid_list)),
             len(np.unique(sesid_list)),
             len(np.unique(logid_list)),
             len(tar), len(non), acllr_all))

        # Create a dict of sessions for each speaker 
        spk2ses_dict = dict()
        for spk in spkid_list:
            # First, get the list of logids for the speaker
            logids_for_spk = [logid for logid, s in logid2spk_dict.items() if s == spk]
            # Then, use the logid2ses dict to get the list of sessions
            spk2ses_dict[spk] = np.unique([ses for logid, ses in logid2ses_dict.items() if logid in logids_for_spk])

        # Create a dict of logids for each session
        ses2lid_dict = dict()
        for ses in logid2ses_dict.values():
            # Get the list of logids for the session
            ses2lid_dict[ses] = [logid for logid, s in logid2ses_dict.items() if s == ses]

        # Lists for accumulating results over each bootstrap sample
        mdcfs, adcfs, acllrs, mcllrs, eers  = [], [], [], [], []

        for nboot in range(num_boot_samples):
            
            # Sample speakers 
            sel_spks = np.array(random.choices(spkid_list, k=len(spkid_list)))

            # Get the list of sessions for the selected speakers
            ses_for_sel_spks = np.unique(np.concatenate([spk2ses_dict[spk] for spk in sel_spks]))

            # Sample those sessions
            sel_sess = np.array(random.choices(ses_for_sel_spks, k=len(ses_for_sel_spks)))

            # Get the list of logids for those sessions
            lid_for_sel_sess = np.unique(np.concatenate([ses2lid_dict[ses] for ses in sel_sess]))

            # Finally, sample those logids or copy the sess id list
            sel_lids = np.array(random.choices(lid_for_sel_sess, k=len(lid_for_sel_sess)) if randomize_by_logid else sel_sess)

            # Get a dict with the correct number of repetitions for each lid.
            # This number is given by the number of times the lid is repeated in 
            # sel_lids, multiplied by the number of times the session for that lid is repeated
            # in sel_sess, multiplied the number of times the speaker for that lid is repeated
            # in sel_spks
            num_rep_dict = dict()
            for lid in logid_list:
                spk = logid2spk_dict[lid]
                ses = logid2ses_dict[lid]
                nlid = len(np.where(sel_lids==lid)[0]) 
                nses = len(np.where(sel_sess==ses)[0]) 
                nspk = len(np.where(sel_spks==spk)[0]) 
                num_rep_dict[lid] = nlid*nses*nspk
            
            # For each test and enroll id, find out how many times
            # it needs to be included in the new key (could be 0, 1 or more)
            def select_indexes(llist):
                sel_idxs = []
                for j, logid in enumerate(llist):
                    sel_idxs += [j]*num_rep_dict[logid]
                return np.array(sel_idxs)

            sel_test_idxs   = select_indexes(key.test_ids)
            sel_enroll_idxs = select_indexes(key.enroll_ids)

            sel_key = Key(np.array(key.test_ids)[sel_test_idxs], np.array(key.enroll_ids)[sel_enroll_idxs], key.mask[np.ix_(sel_enroll_idxs,sel_test_idxs)])
 
            ascores = scores.align(sel_key)
            tar = ascores.score_mat[sel_key.mask==1]
            non = ascores.score_mat[sel_key.mask==-1]
            det = Det(tar, non, sel_key)

            mdcfs.append(det.min_dcf(ptar))
            adcfs.append(det.act_dcf(ptar))
            mcllrs.append(det.min_cllr(ptar))
            acllrs.append(det.act_cllr(ptar))
            eers.append(det.eer()*100)

            print("Sample %d, num_spk = %d, num_sess = %d, num_logids = %d, num_tgt = %d, num_imp = %d, acllr = %.2f"%(nboot, 
                len(np.unique(sel_spks)),
                len(np.unique(sel_sess)),
                len(np.unique(sel_lids)),
                len(tar), len(non), acllrs[-1]))

        def get_conf_int(boot_values):
            return [np.percentile(boot_values, percentile), np.percentile(boot_values, 100-percentile)]

        mdcf_ci  = get_conf_int(mdcfs)
        adcf_ci  = get_conf_int(adcfs)
        mcllr_ci = get_conf_int(mcllrs)
        acllr_ci = get_conf_int(acllrs)
        eer_ci   = get_conf_int(eers)

        outf.write("%-32s |  %6.2f   %6.2f  %6.2f   |  %7.4f  %7.4f  %7.4f  |  %7.4f  %7.4f  %7.4f  | %7.4f %7.4f %7.4f  | %7.4f %7.4f %7.4f  \n"%
            (name, 
            eer_all,   *eer_ci,  
            acllr_all, *acllr_ci,
            mcllr_all, *mcllr_ci,
            adcf_all,  *adcf_ci,
            mdcf_all,  *mdcf_ci))

    return




class Det(object):
    """
    Class to compute different metrics given vectors of target (tar) and impostor (non) scores.
    """
    def __init__(self, tar, non, pav=False):
        
        ntrue  = tar.shape[0]
        nfalse = non.shape[0]
        ntotal = ntrue+nfalse
        if(ntrue==0):
            raise Exception("No target trials found")
        if(nfalse==0):
            raise Exception("No impostor trials found")

        Pmiss  = np.zeros(ntotal+1,np.float32) # 1 more for the boundaries
        Pfa    = np.zeros_like(Pmiss)

        ascores = np.zeros((ntotal,2),np.float32)
        ascores[0:nfalse,0] = non
        ascores[0:nfalse,1] = 0
        ascores[nfalse:ntotal,0] = tar
        ascores[nfalse:ntotal,1] = 1

        ## Sort DET scores.
        # Scores are sorted along the first row while keeping the second row fix (same as MATLAB sortrows)
        ascores = ascores[ascores[:,0].argsort(),]

        sumtrue = np.cumsum(ascores[:,1])
        sumfalse = nfalse - (np.arange(1,ntotal+1)-sumtrue)

        miss_norm = ntrue
        fa_norm = nfalse

        Pmiss[0]  = float(miss_norm-ntrue) / miss_norm
        Pfa[0]    = float(nfalse) / fa_norm
        Pmiss[1:] = (sumtrue+miss_norm-ntrue) / miss_norm
        Pfa[1:]   = sumfalse / fa_norm

        if pav:
            sc = np.concatenate((tar,non))
            la = np.zeros_like(sc,dtype=int)
            la[:len(tar)] = 1.0
            self.pav = calibration.PAV(sc, la)
        else:
            self.pav = None

        self.Pfa = Pfa
        self.Pmiss = Pmiss
        self.scores = ascores[:, 0]
        self.tar = tar
        self.non = non


    def eer(self, from_rocch=False):
        if from_rocch:
            if self.pav is None:
                raise Exception("To extract eer from ROC convex hull you need to call DET with pav=True")
            return calibration.ROCCH(self.pav).EER()

        else:
            idxeer=np.argmin(np.abs(self.Pfa-self.Pmiss))
            return 0.5*(self.Pfa[idxeer]+self.Pmiss[idxeer])


    def act_dcf(self, ptar, normalize=True):
        """
        input:
            ptar: a single value or a vector of target priors 
            normalize: normalize DCFs
        output:
            Values of actualDCF, for optimal thresholds assuming scores are LLR
        """
        ptar = np.asarray(ptar)
        p_non = 1 - ptar
        plo = -1. * np.log(ptar / (1 - ptar))
        dcfs, idxdcfs = np.zeros_like(ptar), np.zeros_like(ptar)

        idx = self.scores.searchsorted(plo)
        dcfs = ptar * self.Pmiss[idx] + p_non * self.Pfa[idx]
        idxdcfs = idx
        if normalize:
            mins = np.amin(np.vstack((ptar, p_non)),axis=0)
            dcfs /= mins

        return dcfs.squeeze()

    def min_dcf(self, ptar, normalize=True):
        """
        input:
            ptar: a single value or a vector of target priors
            normalize: normalize DCFs
        output:
            Values of minDCF, one for each value of ptar
        """
        ptar = np.asarray(ptar)
        p_non = 1 - ptar
        # CDet = CMiss x PTarget x PMiss|Target + CFalseAlarm x (1-PTarget) x PFalseAlarm|NonTarget
        cdet = np.dot(np.vstack((ptar, p_non)).T, np.vstack((self.Pmiss,self.Pfa)))
        idxdcfs = np.argmin(cdet, 1)
        dcfs = cdet[np.arange(len(idxdcfs)), idxdcfs]

        if normalize:
            mins = np.amin(np.vstack((ptar, p_non)), axis=0)
            dcfs /= mins
        return dcfs.squeeze()


    def act_cllr(self, ptar, tar=None, non=None):
        """
        Calculate the CLLR of the scores. This method should give identical results to compute_loss in modules.py, 
        when the type is cross_entropy.
        """
        cllrs = []
        for p in np.atleast_1d(ptar):
            cllrs.append(calibration.cross_entropy(self.tar, self.non, p))

        return np.array(cllrs).squeeze()


    def min_cllr(self, ptar, with_pav=False):

        cllrs = []

        if with_pav:
            if self.pav is None:
                raise Exception("To extract min_cllr using PAV you need to call DET with pav=True")
            
            llrs, ntar, nnon = self.pav.llrs()
            for p in np.atleast_1d(ptar):

                logitPost = llrs + logit(p)

                Ctar, Cnon = calibration.softplus(-logitPost), calibration.softplus(logitPost)
                min_cllr = p*(Ctar[ntar!=0] @ ntar[ntar!=0]) / ntar.sum() +  (1-p)*(Cnon[nnon!=0] @ nnon[nnon!=0]) / nnon.sum()  
                min_cllr /= -p*np.log(p) - (1-p)*np.log(1-p)

                cllrs.append(min_cllr)
        else:

            for p in np.atleast_1d(ptar):
                cal = calibration.logregCal(self.tar, self.non, p)
                tar_cal, non_cal = cal(self.tar), cal(self.non)
                cllrs.append(calibration.cross_entropy(tar_cal, non_cal, p))

        return np.array(cllrs).squeeze()



def create_aligned_mask(orig_mask, target_enroll, target_test, orig_enroll, orig_test):
    """ Rearrange the mask so that the row and columns now correspond to the new
    lists (target_enroll and target_test) instead of the original ones.
    """

    (hase, eindx) = utils.ismember(target_enroll, orig_enroll)
    (hast, tindx) = utils.ismember(target_test, orig_test)
    eindx = eindx[hase]
    tindx = tindx[hast]
    mask = np.zeros([len(target_enroll), len(target_test)], 'f')
    mask[np.ix_(hase, hast)] = orig_mask[np.ix_(eindx, tindx)]

    missing = np.ones([len(target_enroll), len(target_test)], int)
    missing[np.ix_(hase, hast)] = 0

    return mask, hase, hast, missing

