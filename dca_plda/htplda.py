# The code in this file was written by Niko Brummer. 
# Luciana Ferrer added the functionality to (approximately) score with stats.

import torch
from torch.autograd import Function
from torch import as_tensor as ten
import scipy
import numpy as np
torch.backends.cudnn.enabled = False

class ScoreMatrix(Function):
    
    @staticmethod
    def forward(ctx, H, F, nu, X1, X2, Side1=None):
        # As far as I understand the docs, 2nd-order derivatives won't work here.
        # You need to use ctx.save_for_backward() to store all required tensors
        # This would require a refactor to implement each subfunction of
        # score_batch0 as a Function object.

        LLR, ctx.back = score_matrix0(H, F, nu, X1, X2, Side1 = Side1, deriv=True)
        return LLR
    
    @staticmethod
    def backward(ctx,dLLR):
        # ctx.back returns the Side1 gradient, but it does not make sense to backprop
        # this variable
        dH, dF, dnu, dX1, dX2, _ = ctx.back(dLLR)
        return dH, dF, dnu, dX1, dX2, None


score_matrix = ScoreMatrix.apply        
"""
def score_matrix(H, F, nu, X1, X2, Side1=None):
    For X1 containing m x-vectors and X2 containing n, computes the (m,n)
    matrix of LLRs for simple binary trials.
    
    Inputs (model parameters, data, trial specification):
        
        H: (D,D) matrix, used as W = H @ H.T, where W is within-class precision
        F: (D,d) speaker factor loading matrix
        nu: degrees of freedom
        X1: (D,m) batch of x-vectors, m of them, of dimension D
        X2: (D,n) batch of x-vectors, n of them, of dimension D
        Side1: (m,k) flag matrix, with 0/1 entries. Associates the m enrollment
               x-vectors in X1 with k speakers (usually k < m). If omitted, the
               identity matrix is assumed.
    
    Output: 
        
        LLR: (m,n) matrix of scores  
"""


class ScoreMatrixWithStats(Function):
    
    @staticmethod
    def forward(ctx, H, F, nu, M1, X2, N1):
        # Instead of taking X1 and Side1 as input, take the stats for the enrollment
        # models, M1 (mean) and N1 (counts)
        # What we do below is an approximation (see score_matrix_with_stats for more info)
        side1 = torch.diag(N1)
        LLR, ctx.back = score_matrix0(H, F, nu, M1, X2, Side1=side1, deriv=True)
        return LLR
    
    @staticmethod
    def backward(ctx,dLLR):
        dH, dF, dnu, dM1, dX2, dSide1 = ctx.back(dLLR)
        return dH, dF, dnu, dM1, dX2, torch.diag(dSide1)


score_matrix_with_stats = ScoreMatrixWithStats.apply 
"""
def score_matrix(H, F, nu, X1, X2, Side1=None):
    For M1 containing m mean x-vectors, N1 containing the corresponding counts and 
    X2 containing n x-vectors, computes the (m,n) matrix of LLRs for simple binary trials.
    
    Inputs (model parameters, data, trial specification):
        
        H: (D,D) matrix, used as W = H @ H.T, where W is within-class precision
        F: (D,d) speaker factor loading matrix
        nu: degrees of freedom
        M1: (D,m) batch of mean x-vectors, m of them, of dimension D
        N1: (m) counts corresponding to each of the means in M1
        X2: (D,n) batch of x-vectors, n of them, of dimension D
    
    Output: 
        
        LLR: (m,n) matrix of scores  

    Note that this is only an approximation to what really needs to be done.
    Assume that

        N1 = sum(Side1, axis=0)
        M1 = X1 @ Side1

    where X are the original x-vectors for the individual samples. A1 and b1 
    should be given by (see the score_matrix0 method):

    b1 = b @ Side1
    A1 = A @ Side1

    where (see Ab method):

    b = (nu + D-d) / (nu + (X*(G@X)).sum(axis=0))
    A = b * (WF.T @ X)

    Now, if we do not have X1 but only M1 and N1, we cannot compute b1, but we
    can approximate it as:

    b1 = b * N1

    with 

    b = (nu + D-d) / (nu + (M1*(G@M1)).sum(axis=0)) 

    and then 

    A = b * (WF.T @ M1) * N1
"""

def loglikelihoods(Eigen,A,b, deriv=False):
    """
    Computes a vector of log-likelihoods. 
    
    For trial j, A[:,j] and b[j] represent additive statistics computed from 
    one or more x-vectors that are hypothesized to be of the same speaker.
    
    llh[j] is the log-likelihood that these x-vectors belong to the same speaker.
    
    L is used in: B = L @ L.T is a derived HT-PLDA model parameter (a precision matrix)
    
    Inputs:
        
        Eigen: a tuple containing the eigenanalysis of B = LL'
        A: (d,n)
        b: (n,)
        
    Outputs:
        
        llh: (n,) a vector of log-likelihoods
        
        back: optional backprop function handle: dL, dA, db = back(dllh)
        
    """
    
#        B = L @ L.T                            # (d,d)
#        E, V = eigh(B)                         # B = V @ E @ V.T
#        Ecol = E.reshape(-1,1)    

    E, V, Ecol = Eigen

    # P = I + bB
    bE = b*Ecol
    logdets =  torch.log1p(bE).sum(axis=0)  # logdet(P), for every b
    bE1 = 1+bE
    R = (V.T @ A) / bE1
    S =  V @ R      # P\a, for every column a in A    
    y = (A*S).sum(axis=0)
    llh = (y - logdets)/2
    if not deriv: return llh

    def back(dllh):

        dA = dllh*S
        dllhb = dllh*b
        M1 = (S*dllhb)@S.T                        
        M2 = (V*(dllhb/bE1).sum(axis=1))@V.T
        dL = -(M1+M2) # @ L        # defer the multiplication 
        
        #vBv = E*(V**2).sum(axis=0)     # = (V*(B@V)).sum(axis=0)
        #t1 = (vBv.reshape(-1,1) / bE1 ).sum(axis=0) 
        vv = (V**2).sum(axis=0)**2
        t1 = ((vv*E).reshape(-1,1)/bE1 ).sum(axis=0)
        
        t2 = E @ R**2
        db = (-0.5*dllh)*(t1+t2)
        
        
        return dL, dA, db
    return llh, back
        
        

def G_WF_L(H,F,deriv=False):
    """
    Computes some derived model parameters.
    
    Inputs (model parameters):
        H: (D,D), used in W = H @ H.T, where W is within-class precision
        F: (D,d)  D > d, speaker factor loading matrix
        
    Outputs (derived model parameters):
        
        G: (D,D): To be used in x'Gx, to project x-vector energy (squared 
                  magnitude) into complement of speaker subspace. This is used 
                  elsewhere to compute b, the precision scaling factor.
        WF: (D,d) To project x-vector into speaker subspace. This is used
                  elsewehere to compute the first-order stats A.          
                  
        Eigen: a tuple containing the eigenanalysis of B
                  
    """
    W = H @ H.T            # (D,D)
    WF = W @ F             # (D,d)
    B = F.T @ WF           # (d,d) 
    E, V = torch.symeig(B,eigenvectors=True)    # B = V @ E @ V.T
    Ecol = E.reshape(-1,1)
    
    Eigen = E, V, Ecol 
    
    def solveB(RHS):
        return V @ ((V.T @ RHS)/Ecol)
        
    S = solveB(WF.T) 
    G = W - WF @ S      
    L = F.T @ H
    if not deriv: return G, WF, Eigen
    def back(dG,dWF,dL):
    
        # L = F.T @ H
        dL = dL @ L              # perform deferred multiplication
        dF = H @ dL.T
        dH = F @ dL
        
        # G = W - WF @ S
        dW = dG
        dWF = dWF - dG @ S.T    # avoiding side-effect on dWF
        dS = -WF.T @ dG
        
        #S = solveB(WF.T)
        dRHS = solveB(dS)     # B is symmetric
        dWF += dRHS.T
        dB = -dRHS @ S.T
        
        # B = F.T @ WF    ==> B.T = WF.T @ F 
        dWF += F @ dB
        dF += WF @ dB.T
        
        # WF = W @ F
        dW = dW + dWF @ F.T       # avoiding side-effect on dG
        dF += W.T @ dWF
        
        # W = H @ H.T
        dH += (dW+dW.T) @ H
        return dH, dF
    return G, WF, Eigen, back



def Ab(G,WF,nu,X, deriv=False):
    """
    Computes additive statistics (natural parameters) for each of n Gaussian 
    speaker factor (z below) likelihood functions: 
        
        P(X[:,i] | z) \propto N(z | A[:,i]z, (b[i]B)^{-1} )
    
    Inputs: 
        
        G: (D,D), derived model parameter 
        WF: (D,d), derived model parameter
        nu: scalar > 0, model parameter (degrees of freedom)
        
        X: (D,n) a batch of n x-vectors of dimension D
    
    
    Outputs:
        
        A: (d,n)  precisions * means, natural parameters for each of n cases 
        b: (n,) precision scaling factors, for each of n cases 
    
    """
    
    
    D,d = WF.shape
    num = nu + D-d
    GX = G @ X
    q = (X*GX).sum(axis=0)
    den = nu + q
    b = num / den
    P = WF.T @ X
    A = b * P
    if not deriv: return A,b

    def back(dA,db):
        
        # A = b * P
        db = db + (dA*P).sum(axis=0)   # avoid side-effect on db
        dP = b*dA

        # P = WF.T @ X
        dWF = X @ dP.T
        dX = WF @ dP
        
        # b = num / den
        dnum = db / den
        dden = -b*dnum     # -db*num / den**2
        
        # den = nu + q
        dnu = dden.sum()
        dq = dden
        
        # q = (X*GX).sum(axis=0)
        dX += dq*GX
        dGX = dq*X  
        
        #  GX = G @ X
        dG = dGX @ X.T
        dX += G.T @ dGX
        
        # num = nu + D-d
        dnu += dnum.sum()
        
        return dG, dWF, dnu, dX
    return A, b, back
        


def score_matrix0(H, F, nu, X1, X2, Side1=None, deriv=False):
    """
    For X1 containing m x-vectors and X2 containing n, computes the (m,n)
    matrix of LLRs for simple binary trials.
    
    Inputs (model parameters, data, trial specification):
        
        H: (D,D) matrix, used as W = H @ H.T, where W is within-class precision
        F: (D,d) speaker factor loading matrix
        nu: degrees of freedom
        X1: (D,m) batch of x-vectors, m of them, of dimension D
        X2: (D,n) batch of x-vectors, n of them, of dimension D
        Side1: (m,k) flag matrix, with 0/1 entries. Associates the m enrollment
               x-vectors in X1 with k speakers (usually k < m). If omitted, the
               identity matrix is assumed.
    
    Output: 
        
        LLR: (m,n) matrix of scores  
    
    """
    D, m = X1.shape
    n = X2.shape[1]
    k = m if Side1 is None else Side1.shape[1]
    
    if not deriv:        
        G, WF, Eigen = G_WF_L(H,F)
        Ae, be = Ab(G,WF,nu,X1)
        A1, b1 = (Ae, be) if Side1 is None else (Ae @ Side1, be @ Side1) 
        A2, b2 = Ab(G,WF,nu,X2)
        den1 = loglikelihoods(Eigen, A1, b1)
        den2 = loglikelihoods(Eigen, A2, b2)

        b12 = b1.reshape(-1,1) + b2    #(k,n)
        A12 = add_all_cols(A1,A2)  #(D,k,n)

        num = loglikelihoods(Eigen, A12.reshape(-1,k*n), b12.reshape(-1))  #(kn,)
        LLR = num.reshape(k,n) - den1.reshape(-1,1) - den2
        return LLR
    
    G, WF, Eigen, back1 = G_WF_L(H,F,deriv=True)
    Ae, be, back2 = Ab(G,WF,nu,X1,deriv=True)
    A1, b1 = (Ae, be) if Side1 is None else (Ae @ Side1, be @ Side1) 
    A2, b2, back3 = Ab(G,WF,nu,X2,deriv=True)
    den1, back4 = loglikelihoods(Eigen, A1, b1, deriv=True)
    den2, back5 = loglikelihoods(Eigen, A2, b2, deriv=True)

    b12 = b1.reshape(-1,1) + b2    #(k,n)
    A12, back6 = add_all_cols(A1, A2, deriv=True)

    num, back7 = loglikelihoods(Eigen, A12.reshape(-1,k*n), b12.reshape(-1), 
                                deriv=True)
    LLR = num.reshape(k,n) - den1.reshape(-1,1) - den2
    def back(dLLR):
        # LLR = num.reshape(k,n) - den1.reshape(-1,1) - den2
        dnum = dLLR.reshape(-1)
        dden1 = -dLLR.sum(axis=1)
        dden2 = -dLLR.sum(axis=0)
        
        # num, back7 = loglikelihoods(L,A12.reshape(-1,k*n), b12.reshape(-1))
        dL, dA, db = back7(dnum)
        dA12 = dA.reshape(-1,k,n)
        db12 = db.reshape(k,n)

        # A12, back6 = add_all_cols(A1,A2)
        dA1, dA2 = back6(dA12)

        # b12 = b1.reshape(-1,1) + b2
        db1 = db12.sum(axis=1)
        db2 = db12.sum(axis=0)
        
        # den2, back5 = loglikelihoods(Eigen, A2, b2, deriv=True)
        dL_2, dA2_2, db2_2 = back5(dden2)
        dL += dL_2
        dA2 = dA2 + dA2_2
        db2 = db2 + db2_2
        
        # den1, back4 = loglikelihoods(Eigen,A1,b1,deriv=True)
        dL_1, dA1_1, db1_1 = back4(dden1)      
        dL += dL_1
        dA1 = dA1 + dA1_1
        db1 = db1 + db1_1
        
        # A2, b2, back3 = Ab(G,WF,nu,X2,deriv=True)
        dG, dWF, dnu, dX2 = back3(dA2, db2)
        
        # A1, b1 = (Ae, be) if Side1 is None else (Ae @ Side1, be @ Side1)
        dAe, dbe = (dA1, db1) if Side1 is None else (dA1 @ Side1.T, db1 @ Side1.T)
        dSide1 = None if Side1 is None else be.T @ dbe + Ae.T @ dAe 

        # Ae, be, back2 = Ab(G,WF,nu,X1,deriv=True)
        dG_1, dWF_1, dnu_1, dX1 = back2(dAe, dbe)
        dG += dG_1
        dWF += dWF_1
        dnu += dnu_1

        # G, WF, L, back1 = G_WF_L(H,F,deriv=True)
        dH, dF = back1(dG, dWF, dL)
        
        return dH , dF, dnu, dX1, dX2, dSide1
    return LLR, back
    

def add_all_rows(L,R, deriv=False):
    """
    For L(m,d) and R(n,d) return Sum (m,n,d), where:
        Sum[i,j,:] = L[i,:] + R[j,:]
    """
    S = L.unsqueeze(0).transpose(0,1) + R.unsqueeze(0)
    if not deriv: return S
    def back(dS):
        dL = dS.sum(axis=1)
        dR = dS.sum(axis=0)
        return dL, dR
    return S, back


def add_all_cols(L,R, deriv=False):
    """
    For L(d,m) and R(d,n) return Sum (d,m,n), where:
        Sum[:,i,j] = L[:,i] + R[:,j]
    """
    S = L.unsqueeze(-1) + R.unsqueeze(-1).transpose(1,2)
    if not deriv: return S
    def back(dS):
        dL = dS.sum(axis=2)
        dR = dS.sum(axis=1)
        return dL, dR
    return S, back

    
