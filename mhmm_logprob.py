import numpy as np
from mixgauss_prob import mixgauss_prob
from fwdback2 import fwdback2

def mhmm_logprob(data,prior,transmat,mu,Sigma,*mixmat_val):

     # LOG_LIK_MHMM Compute the log-likelihood of a dataset using a (mixture of) Gaussians HMM
     # [loglik, errors] = log_lik_mhmm(data, prior, transmat, mu, sigma, mixmat)
     #
     # data{m}(:,t) or data(:,t,m) if all cases have same length
     # errors  is a list of the cases which received a loglik of -infinity
     #
     # Set mixmat to ones(Q,1) or omit it if there is only 1 mixture component

     if data.ndim < 3:
        ndata = {"0":data}
     else:
        ndata = {"0":data[:,:,0]}
        for i in range(1,len(data[0][0])):
             ndata[str(i)] = data[:,:,i]
     data = ndata

     Q = len(prior)

     if len(mixmat_val) == 0:
         mixmat_val = np.ones(Q)
     else:
         mixmat_val = mixmat_val[0]

     ncases = len(data)

     loglik = 0
     errors = np.array([])
     for m in range(0,ncases):
         obslik = mixgauss_prob(data[str(m)],mu,Sigma,mixmat_val)
         alpha,beta,gamma,ll,s2,g2 = fwdback2(prior,transmat,obslik,fwd_only = 1)
         if ll == -1*np.inf:
              errors = np.append(errors,m)
         loglik = loglik + ll

     return loglik

def main():

     testX = np.loadtxt("testfiles/testXmmhmm.csv",delimiter = ',')
     prior1 = np.loadtxt("testfiles/prior1mmhmm.csv",delimiter = ',')
     transmat1 = np.loadtxt("testfiles/transmat1mmhmm.csv",delimiter = ',')
     mu1 = np.loadtxt("testfiles/mu1mmhmm.csv",delimiter = ',')
     Sigma1 = np.loadtxt("testfiles/Sigma1mmhmm.csv",delimiter = ',')
     mixmat1 = np.loadtxt("testfiles/mixmat1mmhmm.csv",delimiter = ',')
     Sigma1 = np.reshape(Sigma1,(1,1,3,2), order = 'F')
     mu1 = np.reshape(mu1,(1,3,2), order = 'F')

     loglik = mhmm_logprob(testX, prior1, transmat1, mu1, Sigma1, mixmat1)

     print("loglik = ",loglik)

if __name__ == "__main__":
     main()
 
