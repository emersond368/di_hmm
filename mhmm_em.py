import numpy as np
import numpy.matlib
import fwdback2c
import hmmFwdBackc
from em_converged import em_converged
from mixgauss_prob import mixgauss_prob
from fwdback2 import fwdback2 
from normalize import normalize
from mk_stochastic import mk_stochastic
from mixgauss_Mstep import mixgauss_Mstep

def mhmm_em(data, prior, transmat, mu, Sigma, mixmat, verbose = 1,max_iter=10,thresh = 1e-4,cov_type_val = 'full',adj_prior = 1,adj_trans = 1,adj_mix = 1,adj_mu = 1,adj_Sigma =1):

     # LEARN_MHMM Compute the ML parameters of an HMM with (mixtures of) Gaussians output using EM.
     # [ll_trace, prior, transmat, mu, sigma, mixmat] = learn_mhmm(data, ...
     #   prior0, transmat0, mu0, sigma0, mixmat0, ...) 
     #
     # Notation: Q(t) = hidden state, Y(t) = observation, M(t) = mixture variable
     #
     # INPUTS:
     # data[ex][:,t] or data[:,t,ex] if all sequences have the same length
     # prior(i) = Pr(Q(1) = i), 
     # transmat(i,j) = Pr(Q(t+1)=j | Q(t)=i)
     # mu(:,j,k) = E[Y(t) | Q(t)=j, M(t)=k ]
     # Sigma(:,:,j,k) = Cov[Y(t) | Q(t)=j, M(t)=k]
     # mixmat(j,k) = Pr(M(t)=k | Q(t)=j) : set to [] or ones(Q,1) if only one mixture component
     #
     # Optional parameters may be passed as 'param_name', param_value pairs.
     # Parameter names are shown below; default values in [] - if none, argument is mandatory.
     #
     # 'max_iter' - max number of EM iterations [10]
     # 'thresh' - convergence threshold [1e-4]
     # 'verbose' - if 1, print out loglik at every iteration [1]
     # 'cov_type' - 'full', 'diag' or 'spherical' ['full']
     #
     # To clamp some of the parameters, so learning does not change them:
     # 'adj_prior' - if 0, do not change prior [1]
     # 'adj_trans' - if 0, do not change transmat [1]
     # 'adj_mix' - if 0, do not change mixmat [1]
     # 'adj_mu' - if 0, do not change mu [1]
     # 'adj_Sigma' - if 0, do not change Sigma [1]
     #
     # If the number of mixture components differs depending on Q, just set  the trailing
     # entries of mixmat to 0, e.g., 2 components if Q=1, 3 components if Q=2,
     # then set mixmat(1,3)=0. In this case, B2(1,3,:)=1.0.

     previous_loglik = -10000000000#-1*np.inf
     loglik = 0
     converged = 0
     num_iter = 1
     LL = np.array([])
     
     if data.ndim < 3:
        ndata = {"0":data}
     else:
        ndata = {"0":data[:,:,0]}
        for i in range(1,len(data[0][0])):
             ndata[str(i)] = data[:,:,i]
     data = ndata

     numex = len(data)
     
     if data["0"].ndim == 1:
          O = 1
     else:
          O = len(data["0"])
     Q = len(prior)
     if len(mixmat) == 0:
         mixmat = np.ones(Q)

     if mixmat.ndim == 1:
         M = 1
     else:
         M = len(mixmat[0])
     if M == 1:
         adj_mix = 0

     while (num_iter <= max_iter) and (converged == 0):
          # E step
          loglik,exp_num_trans,exp_num_visits1,postmix,m,ip,op = ess_mhmm(prior,transmat,mixmat,mu,Sigma,data)
          #M step
          if bool(adj_prior):
               prior,z = normalize(exp_num_visits1)
          if bool(adj_trans):
               transmat = mk_stochastic(exp_num_trans)
          if bool(adj_mix):
               mixmat = mk_stochastic(postmix)
 
          if adj_mu or adj_Sigma:
               mu2,Sigma2 = mixgauss_Mstep(postmix,m,op,ip,cov_type = cov_type_val)
               if bool(adj_mu):
                    mu = np.reshape(mu2,[O,Q,M],order = 'F')
               if bool(adj_Sigma):
                    Sigma = np.reshape(Sigma2,[O,O,Q,M],order = 'F')
          if bool(verbose):
               print("iteration = ",num_iter," loglik = ",loglik)

          num_iter = num_iter + 1
          converged,decrease = em_converged(loglik,previous_loglik,threshold = thresh)
          previous_loglik = loglik
          LL = np.append(LL,loglik)
     
     return LL,prior,transmat,mu,Sigma,mixmat       
     

def ess_mhmm(prior,transmat,mixmat_val, mu, Sigma, data,verbose=1):

     # ESS_MHMM Compute the Expected Sufficient Statistics for a MOG Hidden Markov Model.
     #
     # Outputs:
     # exp_num_trans(i,j)   = sum_l sum_{t=2}^T Pr(Q(t-1) = i, Q(t) = j| Obs(l))
     # exp_num_visits1(i)   = sum_l Pr(Q(1)=i | Obs(l))
     #
     # Let w(i,k,t,l) = P(Q(t)=i, M(t)=k | Obs(l))
     # where Obs(l) = Obs(:,:,l) = O_1 .. O_T for sequence l
     # Then 
     # postmix(i,k) = sum_l sum_t w(i,k,t,l) (posterior mixing weights/ responsibilities)
     # m(:,i,k)   = sum_l sum_t w(i,k,t,l) * Obs(:,t,l)
     # ip(i,k) = sum_l sum_t w(i,k,t,l) * Obs(:,t,l)' * Obs(:,t,l)
     # op(:,:,i,k) = sum_l sum_t w(i,k,t,l) * Obs(:,t,l) * Obs(:,t,l)' 

     numex = len(data) # number of ex
     if data["0"].ndim == 1:
         O = 1
     else:
         O = len(data["0"])   # rows per ex
     Q = len(prior)
     if mixmat_val.ndim == 1:
          M = 1
     else:
          M = len(mixmat_val[0])
     exp_num_trans = np.zeros((Q,Q))
     exp_num_visits1 = np.zeros(Q)
     postmix = np.zeros((Q,M))
     m = np.zeros((O,Q,M))
     op = np.zeros((O,O,Q,M))
     ip = np.zeros((Q,M))

     mix = (M>1)

     loglik = 0
     if bool(verbose):
          print("forwards-backwards example # ")
     for ex in range(0,numex):
          if bool(verbose):
               print("ex =", ex)
          obs = data[str(ex)]
          if obs.ndim == 1:
               T = len(obs)
          else:
               if obs.shape[1]== 1: #col vector
                   T = len(obs)
               else: #row vector
                   T = len(obs[0])
          if bool(mix):
                B,b2 = mixgauss_prob(obs,mu,Sigma,mixmat_val)
              #  alpha,beta,gamma,current_loglik,xi_summed,gamma2 = fwdback2(prior,transmat,B,obslik2 = b2,mixmat = mixmat_val,compute_xi=1,compute_gamma2 = 1)  # python version
                #b2 = probability at obs given comp k and Hidden state i
                #B = probability at obs given Hidden state i 
                alpha,beta,gamma,current_loglik,gamma2,xi_summed= fwdback2c.MF(prior,transmat,B,M,mixmat=mixmat_val,B2=b2);  # c version
          else:
                B,b2 = mixgauss_prob(obs,mu,Sigma)
              #  alpha,beta,gamma,current_loglik,xi_summed,gamma2 = fwdback2(prior,transmat,B,compute_xi=1) # python version
                alpha,beta,gamma,current_loglik,xi_summed =  fwdback2c.MF(prior,transmat,B,1);# c version
     
          loglik = loglik + current_loglik
          if bool(verbose):
                print("loglik at ",ex," = ",loglik)
          exp_num_trans = exp_num_trans + xi_summed
          exp_num_visits1 = exp_num_visits1 + gamma[:,0]
          if bool(mix):
               postmix = postmix + np.sum(gamma2,axis=2) # sum along 3 axis
          else:
               postmix = postmix + np.sum(gamma,axis=1)
               gamma2 = np.reshape(gamma,(Q,1,T),order = 'F') #gamma2[i][m][t] = gamma[i][t]
          for i in range(0,Q): 
               for k in range(0,M):
                    w = np.reshape(gamma2[i,k,:],(T)) # w[t] = w[i][k][t][l]
                   # wobs = obs * np.matlib.repmat(w,0,1) #wobs[:,t] = w[t] * obs[:,t]
                    wobs = obs * w #fraction of observable resides in hidden state i and mix component k at time t
                    m[:,i,k] = m[:,i,k] + np.sum(wobs,axis=0) # m[:] = sum_t w[t] obs[:,t] unnormalized avg of obs in state i and comp k over time, unnormalized gmm mu
                    op[:,:,i,k] = op[:,:,i,k] + np.dot(wobs, np.transpose(obs)) #obs^2 in state i and comp k over time  
                    ip[i,k] = ip[i,k] + np.sum(np.sum(wobs*obs,axis=0)) #used for gmm sigma, fraction obs^2 at each time in comp k and state i before subtracting mu  
     return loglik,exp_num_trans,exp_num_visits1,postmix,m,ip,op


def main():

#    M = 2
#     testX = np.loadtxt("testfiles/testXmmhmm.csv",delimiter = ',')
#     prior0 = np.loadtxt("testfiles/prior0mmhmm.csv",delimiter = ',')
#     transmat0 = np.loadtxt("testfiles/transmat0mmhmm.csv",delimiter = ',')
#     mu0 = np.loadtxt("testfiles/mu0mmhmm.csv",delimiter = ',')
#     Sigma0 = np.loadtxt("testfiles/Sigma0mmhmm.csv",delimiter = ',')
#     mixmat0 = np.loadtxt("testfiles/mixmat0mmhmm.csv",delimiter = ',')
#     Sigma0 = np.reshape(Sigma0,(1,1,3,2), order = 'F')
#     mu0 = np.reshape(mu0,(1,3,2), order = 'F')

#     LL,prior1,transmat1,mu1,Sigma1,mixmat1 = mhmm_em(testX,prior0,transmat0,mu0,Sigma0,mixmat0,max_iter=500,thresh=1e-5)

#     print "LL = ",LL
#     print "prior1 = ",prior1
#     print "transmat1 = ",transmat1
#     print "mu1 = ",mu1
#     print "Sigma1 = ",Sigma1
#     print "mixmat1 = ",mixmat1
    
     # M = 1
#     testX = np.loadtxt("testfiles/testXmhmm.csv",delimiter = ',')
#     prior0 = np.loadtxt("testfiles/prior0mhmm.csv",delimiter = ',')
#     transmat0 = np.loadtxt("testfiles/transmat0mhmm.csv",delimiter = ',')
#     mu0 = np.loadtxt("testfiles/mu0mhmm.csv",delimiter = ',')
#     Sigma0 = np.loadtxt("testfiles/Sigma0mhmm.csv",delimiter = ',')
#     mixmat0 = np.loadtxt("testfiles/mixmat0mhmm.csv",delimiter = ',')
#     Sigma0 = np.reshape(Sigma0,(1,1,3,1), order = 'F')
#     mu0 = np.reshape(mu0,(1,3,1), order = 'F')
#     mixmat0 = np.reshape(mixmat0,(3,1),order = 'F')

 #    LL,prior1,transmat1,mu1,Sigma1,mixmat1 = mhmm_em(testX,prior0,transmat0,mu0,Sigma0,mixmat0,max_iter=500,thresh=1e-5)

#     print "LL = ",LL
#     print "prior1 = ",prior1
#     print "transmat1 = ",transmat1
#     print "mu1 = ",mu1
#     print "Sigma1 = ",Sigma1
#     print "mixmat1 = ",mixmat1

      #M = 3
      testX = np.loadtxt("testfiles/testX3mhmm.csv",delimiter = ',')
      prior0 = np.loadtxt("testfiles/prior03mhmm.csv",delimiter = ',') 
      transmat0 = np.loadtxt("testfiles/transmat03mhmm.csv",delimiter = ',')
      mu0 = np.loadtxt("testfiles/mu03mhmm.csv",delimiter = ',')
      Sigma0 = np.loadtxt("testfiles/Sigma03mhmm.csv",delimiter = ',') 
      mixmat0 = np.loadtxt("testfiles/mixmat03mhmm.csv",delimiter = ',')
      Sigma0 = np.reshape(Sigma0,(1,1,3,3), order = 'F')
      mu0 = np.reshape(mu0,(1,3,3), order = 'F')

      LL,prior1,transmat1,mu1,Sigma1,mixmat1 = mhmm_em(testX,prior0,transmat0,mu0,Sigma0,mixmat0,max_iter=500,thresh=1e-5)

      print("LL = ",LL)
      print("prior1 = ",prior1)
      print("transmat1 = ",transmat1)
      print("mu1 = ",mu1)
      print("Sigma1 = ",Sigma1)
      print("mixmat1 = ",mixmat1)
   

if __name__ == "__main__":
     main()

