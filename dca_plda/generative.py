# The code for training heavy-tail PLDA in this file was written by Niko Brummer

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
from sklearn import discriminant_analysis
from scipy.special import logit
from scipy.sparse import coo_matrix, dia_matrix
from scipy import linalg
from numpy.linalg import eigh, solve
from scipy.special import gammaln, psi
from numpy.random import randn

from dca_plda import utils 
from dca_plda import data as ddata 
from dca_plda import calibration

def compute_ht_plda_model(X, labels, rank, nu, class_weights, niters=0, quiet=False):
    """ 
    Trains an HTPLDA model, from a database (matrix) of x-vectors, 
    labelled wrt class. A VBEM algorithm is used.
    
    The model parameter, mu is initialized to the data mean and is then further 
    trained along with the other parameters, using VBEM. 
    
    Inputs:
        X: (dim,n)  data in columns (MATLAB style)
        labels: (n,) integer class labels
        rank: integer , 1 <= rank < dim,  class subspace dimension
        nu: degrees of freedom, nu>0  
        niters: number of EM iterations to run
    Outputs: (model parameters and training objective)
        
        mu: (dim,)      x-vector mean
        F:  (dim,rank)  class factor loading matrix
        Cw: (dim,dim)   within-class covariance
        
        obj: (niters, ) training objective
    """
    if class_weights is not None:
        raise Exception("Class weights not yet implemented for HT-PLDA")

    x = X.T
    dim,n = x.shape
    assert len(labels)==n
    assert 1 <= rank <= dim
    L, counts = onehotlabels(labels)
    
    mu = x.mean(axis=1)
    Cw = np.identity(dim)
    F = randn(dim,rank)
    
    obj = np.zeros(niters)
    for i in range(niters):
        mu, F, Cw, obj[i] = iteration_mu(mu,nu,F,Cw,x,L,quiet=quiet)
        if not quiet: print("em iter",i," :",obj[i])
    
    return mu, F, Cw



def compute_2cov_plda_model(x, class_ids, weights, em_its=0, init_ids=None):
    """ Follows the "EM for SPLDA" document from Niko Brummer:
    https://sites.google.com/site/nikobrummer/EMforSPLDA.pdf
    """

    if init_ids is None:
        init_ids = class_ids

    BCov, WCov, GCov, mu, muc, stats = compute_lda_model(x, init_ids, weights)
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
            n_classes = np.sum(stats.cweights[idxs])
            yy_sum = np.linalg.solve(L, n_classes * np.eye(V.shape[0]) + L @ y_hat[:, idxs] @ (y_hat[:, idxs].T * stats.cweights[idxs]))
            R += n*yy_sum
            llk += 0.5 * n_classes * Linv.logdet()
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




def compute_gca_model(scores, mask, embeddings_test, embeddings_enroll, k, ptar=0.1, num_it=3):    
    """ Code to compute a GCA model as proposed in Borgstrom's
    'A Generative Approach to Condition-Aware Score Calibration for Speaker Verification'
    This code needs improvements since it is highly innefficient. This will be done in 
    the near future.
    """


    def Q_func(params, N1, N0, r1, r0, r, rx, rxx, N, D):

        phi_mat = []
        for kk in range(len(params['phii'])):
            phi_mat.append(np.diag(params['phii'][kk]))

        Q1 = - N/2 * (1+ 2*D) * np.log(2*np.pi) + np.log(params['rho']) * sum(N1) 
        Q2 = np.log(1-params['rho']) * sum(N0) + 0.5* sum((N1 + N0) * (2*np.log(params['w']) + np.log(params['lambi']) + np.log(abs(np.linalg.det(phi_mat))))) 
        Q3 = -0.5* sum(params['lambi'] * (r -2* r1 * params['muT'] -2* r0 * params['muN'] + N1 * params['muT'] * params['muT'] + N0 * params['muN'] * params['muN'])) 
        Q4 = -0.5 * sum(sum(params['phii'] *(rxx - 2 * rx * params['theta'] + np.transpose((N1+N0) * np.transpose(params['theta'] * params['theta']))))) 
        Q = Q1 + Q2 + Q3 + Q4

        return Q/N, Q1/N, Q2/N, Q3/N, Q4/N

    def normal_1d(mu, pres, x):
        """ Unidimensional normal with mu mean value and sigma the dispersion """
        """ All three elements must be floats """
        
        lognormal = np.log(np.sqrt(abs(pres)) / np.sqrt(2 * np.pi)) - pres/2 * (x - mu) **2

        return lognormal

    def normal_kd(theta_list, pres_list, x_list):
        """ Multivariate normal of dimension k, the lenght of x  """
        """ The theta is the vector of mean values while sigma is the precision matrix """
        """ All three elements must be lists, the sigma is a list containing the elements in the diagonal, only diagonal allowed """
        x = np.asarray(x_list)
        theta = np.asarray(theta_list)
        diag_array = np.asarray(pres_list)
        pres = np.diag(diag_array)
        k = len(x)
        dif = np.subtract(x, theta)    
        lognormal = np.log(np.sqrt(abs(np.linalg.det(pres))) / np.sqrt( (2 * np.pi) **k)) -1/2 * (np.matmul(dif, np.matmul(pres, dif))) 

        return lognormal

    def ink(k, xn, sn, ln, params):

        wk = params['w'][k]
        thetak = params['theta'][k]
        phik = params['phii'][k]
        muTk = params['muT'][k]
        muNk = params['muN'][k]
        lambk = params['lambi'][k]

        N1 = normal_kd(thetak, phik, xn)
        N2 = normal_1d(muTk, float(lambk), sn)
        N3 = normal_1d(muNk, float(lambk), sn)

        logi = np.log(wk) + N1 + ln *N2 + (1-ln) * N3

        return logi

    def zeta(i_nk):

        n = len(i_nk)
        k = len(i_nk[0])
        z_nk = np.zeros((n,k))


        for nn in range(n):
            i_sum = np.log(sum(np.exp(i_nk[nn])))
        
            z_nk[nn] = np.exp(i_nk[nn] - i_sum)

        return z_nk

    def accumulated_params(z, l, s, x):

        n = len(z)
        k = len(z[0])
        auxn1 = np.zeros((n,k))
        auxn0 = np.zeros((n,k))
        auxr1 = np.zeros((n,k))
        auxr0 = np.zeros((n,k))
        auxr = np.zeros((n,k))

        for nn in range(n):
            auxn1[nn] = z[nn] * l[nn]
            auxn0[nn] = z[nn] * (1 - l[nn])
            auxr1[nn] = z[nn] * l[nn] * s[nn]
            auxr0[nn] = z[nn] * (1 - l[nn]) * s[nn]
            auxr[nn] = z[nn] * s[nn] * s[nn]

        nx = len(x[0])
        auxrx = np.zeros((k,nx))

        for kk in range(k):
            count = np.zeros(nx)
            for nn in range(n):
                count = count + z[nn,kk]*x[nn]
            auxrx[kk] = count

        auxrxx = np.zeros((k,nx))

        for kk in range(k):
            count2 = np.zeros(nx)
            for nn in range(n):
                #print(cuenta, z[nn,kk]*x[nn])
                count2 = count2 + z[nn,kk]*x[nn]*x[nn]
            auxrxx[kk] = count2


        N1 = np.sum(auxn1,axis=0) 
        N0 = np.sum(auxn0,axis=0)
        r1 = np.sum(auxr1,axis=0) 
        r0 = np.sum(auxr0,axis=0) 
        r = np.sum(auxr,axis=0) 
        rx = auxrx 
        rxx = auxrxx

        return N1, N0, r1, r0, r, rx, rxx


    def param_update(params, N1, N0, r1, r0, r, rx, rxx, kappa=0):

        k = len(N1)
        n = len(params['theta'][0])
        muT = np.zeros((k))
        muN = np.zeros((k))
        lambi = np.zeros((k))
        theta = np.zeros((k,n))
        phii = np.zeros((k,n))
        rho = sum(N1) / sum(N1+N0)
        w = (N1+N0) / sum(N1+N0)
        
        for kk in range(k):
            muT[kk] = r1[kk] / N1[kk]
            muN[kk] = r0[kk] / N0[kk]
            lambi[kk] = (N1[kk] + N0[kk]) / (r[kk] -2* r1[kk] * params['muT'][kk] -2* r0[kk] * params['muN'][kk]+ N1[kk] * params['muT'][kk] * params['muT'][kk] + N0[kk] * params['muN'][kk] * params['muN'][kk])
            theta[kk] = rx[kk] / (N1[kk] + N0[kk] + kappa)
            phii[kk] = np.divide((N1[kk] + N0[kk]) * np.ones(n), rxx[kk] - 2 * rx[kk] * params['theta'][kk] + (N1[kk] + N0[kk])*  (params['theta'][kk] * params['theta'][kk]))

        return {'rho': rho, 'w': w, 'muT': muT, 'muN': muN, 'lambi': lambi, 'theta': theta, 'phii': phii}

    # Initial values
    params = {}
    dim_emb = embeddings_test.shape[1]
    params['nx'] = 2 * dim_emb
    params['w'] = np.random.uniform(0,1,k)
    params['muT'] = np.ones(k)
    params['muN'] = np.ones(k) * -1
    params['lambi'] = np.ones(k) * 0.01
    params['rho'] = ptar
    phi_n = np.ones(params['nx']) * 0.0003
    params['phii'] = np.repeat([phi_n], k, axis=0)
    theta_Sigma = np.diag(np.ones(params['nx'])*0.01)
    theta_mu = np.zeros(params['nx']) 
    params['theta'] = np.random.multivariate_normal(theta_mu, theta_Sigma, size=k)

    # The training code, for now, takes a list of scores, and a list of concatenated embeddings
    # We should eventually change it to work like the calibrate method which accepts a matrix of
    # scores and the embedding for each side of the trial separately (assuming the phi matrix is diagonal)

    valid_scores = mask!=0
    flat_scores = scores[valid_scores]
    # Labels need to be 0 for impostors, 1 for targets, while the mask uses -1 and 1
    labels = (mask[valid_scores] + 1)/2

    num_samples = len(labels)

    shape_t = embeddings_test.shape
    shape_e = embeddings_enroll.shape
#    e = embeddings_enroll.unsqueeze(1).expand([-1, embeddings_test.shape[0], -1])
#    t = embeddings_test.unsqueeze(0).expand([embeddings_enroll.shape[0], -1, -1])
    target_shape = [shape_e[0], shape_t[0], shape_e[1]]
    e = np.broadcast_to(embeddings_enroll[:,None,:], target_shape)
    t = np.broadcast_to(embeddings_test[None,:,:],   target_shape)
    concat_embeddings = np.concatenate([e, t], 2)[valid_scores]

    for j in range(num_it):
        
        # This double for loop should be done as a single matrix operation
        i_array = np.zeros((num_samples,k))
        for nn in range(num_samples):
            for kk in range(k):
                i_array[nn,kk] = ink(kk, concat_embeddings[nn], flat_scores[nn], labels[nn], params)
                            
        zetank = zeta(i_array)

        N1, N0, r1, r0, r, rx, rxx = accumulated_params(zetank, labels, flat_scores, concat_embeddings)

        params = param_update(params, N1, N0, r1, r0, r, rx, rxx)

        Q, q1, q2, q3, q4 = Q_func(params, N1, N0, r1, r0, r, rx, rxx, num_samples, dim_emb)
        print('GCA iteracion %d, Q = %f'%(j, Q))

    # Keep only the first num_dim for theta y phii since for scoring we now apply the normal for enrollment
    # and test embeddings separately (and there is no reason to believe the transform should be different
    # for enrollment and test sides since scoring is symmetric).
    params['phii']  = params['phii'][:,0:dim_emb].T
    params['theta'] = params['theta'][:,0:dim_emb].T

    return np.log(params['w']), params['theta'], params['phii'], params['muT'], params['muN'], params['lambi']



def compute_stats(x, class_ids, class_weights, sample_weights):
    # Zeroth, first and second order stats per class, considering that
    # each class is weighted by the provided weights
    
    stats     = utils.AttrDict(dict())
    nsamples  = class_ids.shape[0]
    classmat  = coo_matrix((np.ones(nsamples), (class_ids, np.arange(nsamples)))).tocsr()
    stats.cweights = np.atleast_2d(class_weights).T
    stats.sweights = np.atleast_2d(sample_weights).T / np.sum(sample_weights) * float(nsamples)
    sweightsdiag   = dia_matrix((stats.sweights.squeeze(),0),shape=(nsamples,nsamples))
    classmat_sw    = classmat.dot(sweightsdiag) # Matrix with one row per class and one col per sample, with the weights as entries

    # Global mean, considering that each class's data should be 
    # multiplied by the corresponding weight for that class
    sample_weights = stats.cweights[class_ids] * stats.sweights
    stats.mu       = np.array(sample_weights.T @ x) / np.sum(sample_weights) 

    # N and F are stats per class, weights are not involved in the computation of those. 
    # S, on the other hand, is already a sum over classes so the sample from each class
    # has to be weighted by the corresponding weight
    xcent     = x - stats.mu
    xcentw    = xcent * sample_weights
    stats.N   = np.array(classmat_sw.sum(1)) 
    stats.F   = np.array(classmat_sw @ xcent)
    stats.S   = xcent.T @ xcentw

    return stats


def compute_lda_model(x, class_ids, weights):

    class_weights  = weights['class_weights']
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
    
    # Matrix with the weights per class in the diagonal
    n_classes = stats.N.shape[0]
    weightsdia = dia_matrix((class_weights,0),shape=(n_classes, n_classes))

    # BCov and GCov are computed by weighting the contribution of each class with 
    # the provided weights
    Ntot  = np.sum(np.multiply(stats.N, stats.cweights))
    Fs    = stats.F / np.sqrt(stats.N) 
    BCov  = Fs.T @ (Fs * stats.cweights) / Ntot
    GCov  = stats.S / Ntot
    WCov  = GCov - BCov

    return BCov, WCov, GCov, mu.squeeze(), muc, stats



### Utilities for HT-PLDA model training

class RGSDict(dict):
    """
    Utility class typically not directly invoked by the user. 
    """
    def __init__(self):
        dict.__init__(self)
        self.count = 0
    def __getitem__(self,key):    
        if not self.__contains__(key):
            count = self.count
            self.__setitem__(key,count)
            self.count = count + 1
            return count
        else:
            return dict.__getitem__(self,key)


def rgs(labels,return_map=False):
    """
    Maps a sequence of labels to restricted growth string (RGS) format. 
    The first label is mapped to 0, the next new label to 1 and so on. 
    
    Usage:
        
            rgs_labels = rgs(nonrgs_labels)
            
    or        
            
            rgs_labels, map = rgs(nonrgs_labels, return_map=True)
            
    where map is a dict that maps the original labels to the rgs_labels.        
    """
    map = RGSDict()
    rgs = np.array([map[label] for label in labels])
    if return_map: return rgs, map
    return rgs        


def onehotlabels(labels,dtype=int):
    """
    labels: (n,) integer class labels
    
    returns: 
         
         (m,n) matrix, with 0/1 entries, where m is the number of blocks in 
         the partition and n is the number of elements in the partitioned set.
         Columns are one-hot. If return_matrix[i,j]==1, then element j is in 
         block i.
         
         counts: (m,): the number of occurrences for each class
         
    """
    labels = rgs(labels)        
    uu, counts = np.unique(labels,return_counts=True)
    m, n = len(counts), len(labels)
    assert uu[0] == 0 and uu[-1] == m-1, "labels must be in RGS format"
    L = coo_matrix((np.ones(n,dtype=dtype),(labels,np.arange(n))),shape=(m,n)).tocsc()
    return L, counts


    
def iteration_mu(mu,nu,F,Cw,X0,labels,quiet=False):
    """    
    Iteration of VB algorithm for HT-PLDA training. 
    The model parameters mu, F and Cw are updated. 
    
    Inputs:
       nu: scalar, df > 0 (nu=inf is allowed: it signals Gaussian PLDA)
       F: (D,d) factor loading matrix, D > d
       Cw: within-class covariance, (D,D), pos. def,
       X0: (D,N) data matrix (not centered)
       labels: (M,N), one hot columns, labels for M classes and N x-vectors
               This is a large matrix and it is best represented in sparse format
    
    Returns:
        mu, F, Cw : updated versions on the model parameters
        obj: the VB lower bound for this iteration 
    
    """

    scaling_mindiv = True
    Z_mindiv = True

    D,d = F.shape
    M,N = labels.shape
   
    
    W = utils.CholInv(Cw)
    
   
    # E-step
    X = X0 - mu.reshape(-1,1)
    P = (W @ F).T       # (d,D)
    B0 = P @ F          # (d,d) common precision (up to scaling)
    L,V = eigh(B0)      # eigendecomposition B0 = V @ diag(L) @ V.T; V is orthonormal  
    VP = V.T @ P
    
    if np.isinf(nu):
        b = np.full(N,1.0)
    else:
        GX = W @ X - (VP.T/L) @ (VP @ X)
        q = ( X * GX ).sum(axis=0)         # (N,)
        b = (nu+D-d)/(nu+q)                     # (N,)

    
    bX = b * X
    S = bX @ X.T                         # (D,D)  weighted 2nd-order stats 

    f = bX @ labels.T                    # (D,M)  weighted 1st-order stats
    n = b @ labels.T                     # (M,)  weighted 0-order stats
    tot_n = n.sum()
    
    logLPP = np.log1p(L.reshape(-1,1) * n)  # (d,M) log eigenvalues of posterior precisions
    LPC = np.exp(-logLPP)                   # (d,M) eigenvalues of posterior covariances
    logdetPP = logLPP.sum(axis=0)           # logdets of posterior precisions
    tracePC = LPC.sum(axis=0)               # and the traces
    Z = V @ (LPC*(VP@f))                    # (d,M) posterior means
    T = Z@f.T                               # (d,D)
    
    R = (Z*n) @ Z.T + V @ ((LPC@n).reshape(-1,1)*V.T)
    C = ( Z@Z.T + V @ (LPC.sum(axis=1,keepdims=True)*V.T) ) / M
    
    
    
    logdetW = W.logdet()
    trWS = np.trace(W @ S)
    logLH = (N/2)*logdetW + (D/2)*sum(np.log(b)) - 0.5*trWS \
             + trprod(T,P) -0.5*trprod(B0,R)
    
    if np.isinf(nu):
        obj = logLH - KLGauss(logdetPP,tracePC,Z)
    else:
        obj = logLH - KLGauss(logdetPP,tracePC,Z) - KLgamma(nu,D,d,b)
    
    
    
    # M-step
    
    # adapt mu
    Delta = X0 - F @ (Z @ labels)
    mu = (Delta @ b) / b.sum()
    
    # re-center
    X = X0 - mu.reshape(-1,1)
    bX = b * X
    S = bX @ X.T                         # (D,D)  weighted 2nd-order stats 
    f = bX @ labels.T                    # (D,M)  weighted 1st-order stats
    T = Z@f.T                               # (d,D)
    
    
    F = rsolve(T.T,R)
    FT = F @ T
    
    
    if scaling_mindiv:
        Cw = (S - (FT+FT.T)/2)/tot_n
    else:
        Cw = (S - (FT+FT.T)/2)/N
    
    if Z_mindiv:
        Zmean = Z.mean(axis=1)
        Z0 = Z - Zmean.reshape(-1,1)
        C = ( Z0@Z0.T + V @ (LPC.sum(axis=1,keepdims=True)*V.T) ) / M
        Ci = utils.CholInv(C)
        F = F @ Ci.L
        mu += F @ Zmean 
    else:
        C = ( Z@Z.T + V @ (LPC.sum(axis=1,keepdims=True)*V.T) ) / M
        Ci = utils.CholInv(C)

    if not quiet: 
        print("  cov(z): trace =",np.trace(C)," logdet =",-Ci.logdet())
    
    return mu, F, Cw, obj



def rsolve(LHS,A):
    return solve(A.T,LHS.T).T

def trprod(A,B):
    return (A*B).sum()


def KLGauss(logdets,traces,Means):
    d = Means.shape[0]
    M = len(logdets)
    kl = ( traces.sum() - logdets.sum() + trprod(Means,Means) - d*M)/2
    return kl

def KLgamma(nu,D,d,lambdahat):

    # prior has expected value a0/b0 = 1
    a0 = nu/2
    b0 = nu/2
    
    #Posterior
    a = (nu + D - d) / 2
    b = a / lambdahat
    #This value for a is a thumbsuck: mean-field VB gives instead a =(nu+D)/2, 
    #while b is chosen to give lambdahat = a/b. 
    #This gives a larger variance (a/b^2) than the mean-field would, which
    #is probbaly a good thing, since mean-field is known to underestimate variances. 
    
    kl = (gammaln(a0) - gammaln(a) + a0*np.log(b/b0) + psi(a)*(a-a0) + a*(b0-b)/b).sum()
    
    return kl


