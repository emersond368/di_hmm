import numpy as np
from normalize import normalize

def hmmFilter(initDist,transmat,softev):

     # Calculate p(S(t)=i| y(1:t))
     # INPUT:
     # initDist(i) = p(S(1) = i)
     # transmat(i,j) = p(S(t) = j | S(t-1)=i)
     # softev(i,t) = p(y(t)| S(t)=i)
     #
     # OUTPUT
     # loglik = log p(y(1:T))
     # alpha(i,t)  = p(S(t)=i| y(1:t))

     # This file is from pmtk3.googlecode.com

     transmat = np.array(transmat,dtype = float)

     K = len(softev)
     T = len(softev[0])

     scale = np.zeros(T)
     AT = transmat.T

     alpha = np.zeros((K,T))
     alpha[:,0],scale[0] = normalize(initDist[:]*softev[:,0])
     for t in range(1,T):
          alpha[:,t],scale[t] = normalize(np.dot(AT,alpha[:,t-1])*softev[:,t])

     loglik = np.sum(np.log(scale + np.finfo(float).eps))

     return loglik,alpha

    
