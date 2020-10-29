"""
Code for linear logistic regression and pav calibration.

This code is only used to compute cllr and min_cllr in the Det class.
"""

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logit, expit
from IPython import embed
from sklearn._isotonic import _inplace_contiguous_isotonic_regression as fastpav


log2 = np.log(2)

def cross_entropy(tar,non,Ptar=0.5,deriv=False):

    baseline = -Ptar*np.log(Ptar) - (1-Ptar)*np.log(1-Ptar)
    logitprior = logit(Ptar)
    if not deriv:
        t = np.mean(softplus(-tar-logitprior))
        n = np.mean(softplus(non+logitprior))
        return ( Ptar*t  + (1-Ptar)*n ) / baseline
    
    t, back1 = softplus(-tar-logitprior,deriv=True)
    n, back2 = softplus(non+logitprior,deriv=True)
    k1 = Ptar/(len(t)*baseline)
    k2 = (1-Ptar)/(len(n)*baseline)
    y = k1*t.sum()  + k2*n.sum()
    def back(dy):
        dtar = back1(-dy*k1)
        dnon = back2(dy*k2)
        return dtar, dnon
    return y, back


def cs_softplus(x):
    """numerically stable and complex-step-friendly version of: 
       
       softplus = log( 1 + exp(x) )
    """
    if not np.iscomplexobj(x): 
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
    #return np.log( 1 + np.exp(x) )
    rx = np.real(x)
    y = cs_softplus(rx)
    return y + 1.0j*expit(rx)*np.imag(x)

def cs_sigmoid(x):
    """numerically stable and complex-step-friendly version of sigmoid"""
    if not np.iscomplexobj(x): return expit(x)
    rx = np.real(x)
    p, q = expit(rx), expit(-rx)
    return p + 1.0j*p*q*np.imag(x)


def softplus(x, deriv=False):
    y = cs_softplus(x)
    if not deriv: return y
    
    dydx = cs_sigmoid(x)
    def back(dy): return dy*dydx
    return y, back 


def logregCal(tar,non,Ptar=0.5, return_params=False):

    # logregCal requires float64
    tar = np.array(tar,dtype=np.float64)
    non = np.array(non,dtype=np.float64)

    params0 = np.array([1.0,0.0])
    trobj = optobjective(obj,sign=1.0,tar=tar,non=non,Ptar=Ptar)
    res = minimize(trobj,params0,method='L-BFGS-B', jac=True) 
    params = res.x
    a,b = params
    if return_params:
        return a, b
    else:
        return lambda s: trans(a,b,x=s)


def optobjective(f,trans=None,sign=1.0,**kwargs):
    """Wrapper for f to turn it into a minimization objective.
    
    The objective is: 
        obj(x) = sign*f(*trans(x),**kwargs) 
    where f can take multiple inputs and has a scalar output. 
    The input x is a vector.
    
    Both f(...,deriv=True) and trans(...) return backpropagation 
    handles, but obj(x) immediately returns value, gradient.  
    
       
    """
    if not trans is None:
        def obj(x):
            *args, back1 = trans(x, deriv=True)
            y, back2 = f(*args,deriv=True,**kwargs)
            g = back1(*back2(sign))
            return sign*y, g
    else:
        def obj(x):
            y, back2 = f(x,deriv=True,**kwargs)
            g = back2(sign)
            return sign*y, g

    return obj
    
def trans(a,b,*,x,deriv=False):
    if not deriv:
        return a*x + b
    def back(dy):
        da = dy @ x
        db = dy.sum()
        return da, db
    return a*x + b, back    
        
        
def obj(params,*,tar,non,Ptar=0.5,deriv=False):
    a, b = params
    if not deriv: 
        return cross_entropy(trans(a,b,x=tar),trans(a,b,x=non),Ptar=Ptar)
    t, back1 = trans(a,b,x=tar,deriv=True)
    n, back2 = trans(a,b,x=non,deriv=True)
    y, back3 = cross_entropy(t,n,Ptar=Ptar,deriv=True)
    def back(dy):
        dt, dn = back3(dy)
        da, db = back2(dn)
        da2, db2 = back1(dt)
        return np.array([da + da2, db + db2])
    return y, back


class PAV:
    """
    The constructor PAV(scores,labels) invokes the PAV algorithm and stores 
    the result in a convenient form in the newly created object. There are methods 
    to recover the score to log-likelihood-ratio transform and the ROC convex 
    hull.
    """
    def __init__(self,scores,labels):
        """
        Constructor for PAV.
        
            scores: (n,) array of real-valued binary classifier 
                    scores, where more positive scores support class 1, and 
                    more negative scores support class 0. The scores can be 
                    in the form of (possibly uncalibrated) probabilities,
                    likelihood-ratios, or log-likelihood-ratios, all of which
                    satisfy the above criteria for being binary classifier 
                    scores.
            labels: (n,) array of 0/1 labels, identifying the true class 
                    associated with each score. There has to be at least one
                    example of each class.
        """
       
        scores = np.array(scores, dtype=np.float64)
        self.T = T = labels.sum()
        self.N = N = len(labels) - self.T
        assert T > 0 < N
        
        ii = np.lexsort((-labels,scores))  # when scores are equal, those with 
                                           # label 1 are considered inferior
        weights = np.ones_like(scores)
        y = np.empty_like(scores)
        y[:] = labels[ii]
        fastpav(y,weights)

        p, ff, counts = np.unique(y, return_index = True, return_counts = True)
        self.nbins = len(p)
        self.p = p
        self.counts = counts
        self.scores = np.empty((self.nbins,2))
        self.scores[:,0] = scores[ii[ff]]            # bin low scores
        self.scores[:,1] = scores[ii[ff+counts-1]]   # bin high scores

        self.targets = np.rint(counts * p).astype(int)    # number of target scores in each PAV block    
        self.nontars = counts - self.targets              # number of non-tar scores in each PAV block
        assert self.targets.sum() == T
        assert self.nontars.sum() == N

        
    def rocch(self, asmatrix = False):
        """
        This returns the convex hull of the ROC (receiver operating curve)
        associated with the scores and labels that were used to construct this 
        PAV object. The convex hull is returned as a list of vertices in the 
        (Pmiss, Pfa) ROC plane. The Pmiss values are non-decreasing, while the 
        Pfa values are non-increasing. The first vertex is at (0,1) and the 
        last at (1,0). There are always at least two vertices. The vertices 
        describe a convex, piece-wise linear curve. 
        
        The result can be retuned as a (2,n-vertices) matrix, or Pmiss and 
        Pfa can be returned separately, according to the asmatrix flag.
        
        The user does not typically have to invoke this call. Instead the user 
        would do my_rocch = ROCCH(PAV(scores,labels)) and then call some methods
        on my_rocch.
        """
        nbins = self.nbins
        PmissPfa = np.empty((2,nbins+1)) 
        pmiss = PmissPfa[0,:]
        pfa = PmissPfa[1,:]
        pmiss[0] = 0.0
        np.cumsum(self.targets, out = pmiss[1:])
        pmiss /= self.T

        pfa[0] = 0.0
        np.cumsum(self.nontars, out = pfa[1:])
        pfa -= self.N
        pfa /= (-self.N)
        return PmissPfa if asmatrix else (pmiss, pfa)
    
    def llrs(self):
        """
        Returns three vectors: 
            - The llrs for each PAV bin.
            - The number of targets (class 1) in each bin
            - The number of non-targets (class 0) in each bin
        """
        llr = logit(self.p) - np.log(self.T / self.N)
        return llr, self.targets, self.nontars
    
    def scores_vs_llrs(self):
        p = self.p
        LLRs = np.empty_like(self.scores)
        llr = LLRs[:,0]
        llr[:] = logit(p)
        llr -= np.log(self.T / self.N)   
        LLRs[:,1] = llr 
        return self.scores.ravel(), LLRs.ravel()


class ROCCH:
    """
    An ROCCH object contains the convex hull of the ROC (receiver operating 
    curve) associated with a set of bianry classifier scores and labels. This 
    object can be constructed from a PAV object.
    
    """
    def __init__(self,pava):
        """
        ROCCH(PAV(scores,labels)) constructs an object of this class.
        """    
        self.PmissPfa = pava.rocch(True)

    def Pmiss_Pfa(self):
        """
        Returns the set of ROCCH vertices as two (n-vertices,) arrays, for
        each of Pmiss and Pfa.
        """
        return self.PmissPfa[0,:], self.PmissPfa[1,:]

    def Bayes_error_rate(self,prior_log_odds,return_Pmiss_Pfa = False):
        """
        Returns the optimal Bayes error-rate at one or more operating points
        of a binary classifier. The accuracy of the binary classifier is 
        empricially represented by the scores and labels that were used to
        construct this ROCCH object. The `prior' is a synthetic prior 
        probability for class 1, but is passed to this method in log-odds form,
        i.e. prior_log_odds = log(prior) - log(1-prior). One or more prior values
        can be passed, where each value is referred to as an operating point.
        For each operating point at prior = p, the returned optimal 
        Bayes-error-rate is:
            
            BER(p) = min_t p * Pmiss(t) + (1-p) * Pfa(t)
            
        where t is the decision threshold applied to the scores that were used 
        to construct this ROCCH. 
        
        The algorithm used here is typically faster than a naive implementation
        that operates directly on the scores, because the minimization can be 
        done only over the ROCCH vertices, of which are typically far fewer 
        than the original scores.
            
            
        """
        sc = np.isscalar(prior_log_odds)
        m = 1 if sc else len(prior_log_odds)
        PP = np.empty((2,m))
        PP[0,:] = expit(prior_log_odds)
        PP[1,:] = expit(-prior_log_odds)
        E = PP.T @ self.PmissPfa              # (m,nbins+1)
        if not return_Pmiss_Pfa:
            ber = E.min(axis=1)
            return ber.item() if sc else ber
        jj = E.argmin(axis=1)
        ber = E[range(len(jj)),jj]
        Pmiss = self.PmissPfa[0,jj]
        Pfa = self.PmissPfa[1,jj]
        if sc: ber, Pmiss, Pfa = ber.item(), Pmiss.item(), Pfa.item() 
        return ber, Pmiss, Pfa
        
    def EER(self):
        """Returns the value of the equal error-rate (EER), which is equal to 
           the Pmiss and Pfa values on the ROCCH curve, where they are equal.
           The EER is equal to the maximum of the optimal Bayes-error-rate of 
           this curve, when maximized w.r.t. the operating point. 
        
        """
        f = lambda x: -self.Bayes_error_rate(x)
        res = minimize_scalar(f)
        return -f(res.x)
        


if __name__ == "__main__":
    print("Running test script for module logregcal\n")
    
    from numpy.random import rand, randn
    
    tar, non, a, b, Ptar = randn(5), randn(10), randn()**2, randn(), rand()
    
    cal = logregCal(tar,non)
    embed()
    print(cal)

