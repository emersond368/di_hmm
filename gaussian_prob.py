import numpy as np
from numpy.linalg import inv


def gaussian_prob(x,m,C,use_log=0):
     # GAUSSIAN_PROB Evaluate a multivariate Gaussian density
     # p = gaussian_prob(X, m, C)
     # p[i] = N[X[:,i],m,C] where C = covariance matrix and each COLUMN of x is a datavector

     # p = gaussian_prob(X,m,C,1) returns log N[X[:,i],m,c] (to prevents underflow).
     #
     # If X has size dXN,then p has size Nx1, where N = number of examples

     if x.ndim == 1:
          d = 1
          N = len(x)
     else:
          d = len(x)
          N = len(x[0])
    
     M = m*np.ones(N) #replicate the mean across columns
     denom = np.power((2*np.pi),(float(d)/2))*np.sqrt(np.abs(np.linalg.det(C)))
     val = inv(C)
     while val.ndim > 0:
        if val.shape[0] == 1:
            val = val[0]
        else:
             break
     mahal = (np.dot((x-M).T,val)*(x-M).T)
     if mahal.ndim > 1: 
          mahal = np.sum((np.dot((x-M).T,val)*(x-M).T),axis = 1) #Chris Bregler's trick
     if not np.all(mahal >= 0):
          print("WARNING: mahal < 0 => C is not psd")
     if bool(use_log):
          p = -0.5*mahal - np.log(denom)
     else:
          p = np.exp(-0.5*mahal)/(denom + np.finfo(float).eps)
            
     return p
