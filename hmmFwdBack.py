import numpy as np
from normalize import normalize
from hmmFilter import hmmFilter

def hmmFwdBack(initDist,transmat,softev):

     #Calculate p(S(t)=i | y(1:T))
     # INPUT:
     # initDist(i) = p(S(1) = i)
     # transmat(i,j) = p(S(t) = j | S(t-1)=i)
     # softev(i,t) = p(y(t)| S(t)=i)
     #
     # OUTPUT
     # gamma(i,t) = p(S(t)=i | y(1:T))
     # alpha(i,t)  = p(S(t)=i| y(1:t))
     # beta(i,t) propto p(y(t+1:T) | S(t=i))
     # loglik = log p(y(1:T))

     # This file is from pmtk3.googlecode.com


     # Matlab Version by Kevin Murphy
     # C Version by Guillaume Alain
     #PMTKauthor Guillaume Alain
     #PMTKmex

     loglik,alpha = hmmFilter(initDist,transmat,softev)
     beta = hmmBackwards(transmat,softev)
     gammaM,gammaz = normalize(alpha*beta,1) #make each colum sum to 1

     return gammaM,alpha,beta,loglik

def hmmBackwards(transmat,softev):

     K = len(softev)
     T = len(softev[0])

     beta = np.zeros((K,T))
     beta[:,T-1] = np.ones(K)
     for t in range(T-2,-1,-1):
          input1 = beta[:,t+1]*softev[:,t+1]
          valsM,valsz = normalize(np.dot(transmat,input1))
          beta[:,t] = valsM
     return beta

def main():

     prior1 = np.loadtxt("testfiles/prior1.csv",delimiter = ',')
     transmat1 = np.loadtxt("testfiles/transmat1.csv",delimiter = ',')
     B = np.loadtxt("testfiles/Bval.csv",delimiter = ',')

     gamma,alpha,beta,logp = hmmFwdBack(prior1,transmat1,B)

     print("gamma = ",gamma)
     print("alpha = ",alpha)
     print("beta = ",beta)
     print("logp = ",logp)

     prior1 = np.loadtxt("testfiles/prior1mmhmm.csv",delimiter = ',')
     transmat1 = np.loadtxt("testfiles/transmat1mmhmm.csv",delimiter = ',')
     B = np.loadtxt("testfiles/B_mmhm_mgp_outer.csv",delimiter = ',')

     gamma,alpha,beta,logp = hmmFwdBack(prior1,transmat1,B)

     print("gamma = ",gamma)
     print("alpha = ",alpha)
     print("beta = ",beta)
     print("logp = ",logp)

if __name__ == "__main__":
     main()

