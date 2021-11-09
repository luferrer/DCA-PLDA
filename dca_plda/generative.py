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



