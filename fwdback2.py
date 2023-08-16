import numpy as np
import numpy.matlib
from normalize import normalize

def fwdback2(init_state_distrib,transmat,obslik,compute_xi=0,compute_gamma2=0,obslik2 = np.array([]),mixmat = np.array([]),fwd_only = 0,scaled = 1,act = np.array([]),maximize = 0):

     #FWDBACK Compute the posterior probs. in an HMM using the forwards backwards algo.
     #
     #[alpha, beta, gamma, loglik, xi, gamma2] = fwdback(init_state_distrib, transmat, obslik, ...)
     #
     #Notation:
     #Y(t) = observation, Q(t) = hidden state, M(t) = mixture variable (for MOG outputs)
     #A(t) = discrete input (action) (for POMDP models)
     #
     #INPUT:
     #init_state_distrib(i) = Pr(Q(1) = i)
     # transmat(i,j) = Pr(Q(t) = j | Q(t-1)=i)
     #  or transmat{a}(i,j) = Pr(Q(t) = j | Q(t-1)=i, A(t-1)=a) if there are discrete inputs
     # obslik(i,t) = Pr(Y(t)| Q(t)=i)
     #   (Compute obslik using eval_pdf_xxx on your data sequence first.)
     #
     # Optional parameters may be passed as 'param_name', param_value pairs.
     # Parameter names are shown below; default values in [] - if none, argument is mandatory.
     #
     # For HMMs with MOG outputs: if you want to compute gamma2, you must specify
     # 'obslik2' - obslik(i,j,t) = Pr(Y(t)| Q(t)=i,M(t)=j)  []
     # 'mixmat' - mixmat(i,j) = Pr(M(t) = j | Q(t)=i)  []
     #  or mixmat{t}(m,q) if not stationary
     #
     # For HMMs with discrete inputs:
     # 'act' - act(t) = action performed at step t
     #
     # Optional arguments:
     # 'fwd_only' - if 1, only do a forwards pass and set beta=[], gamma2=[]  [0]
     # 'scaled' - if 1,  normalize alphas and betas to prevent underflow [1]
     # 'maximize' - if 1, use max-product instead of sum-product [0]
     #
     # OUTPUTS:
     # alpha(i,t) = p(Q(t)=i | y(1:t)) (or p(Q(t)=i, y(1:t)) if scaled=0)
     # beta(i,t) = p(y(t+1:T) | Q(t)=i)*p(y(t+1:T)|y(1:t)) (or p(y(t+1:T) | Q(t)=i) if scaled=0)
     # gamma(i,t) = p(Q(t)=i | y(1:T))
     # loglik = log p(y(1:T))
     # xi(i,j,t-1)  = p(Q(t-1)=i, Q(t)=j | y(1:T))  - NO LONGER COMPUTED
     # xi_summed(i,j) = sum_{t=}^{T-1} xi(i,j,t)  - changed made by Herbert Jaeger
     # gamma2(j,k,t) = p(Q(t)=j, M(t)=k | y(1:T)) (only for MOG  outputs)
     # If fwd_only = 1, these become
     # alpha(i,t) = p(Q(t)=i | y(1:t))
     # beta = []
     # gamma(i,t) = p(Q(t)=i | y(1:t))
     # xi(i,j,t-1)  = p(Q(t-1)=i, Q(t)=j | y(1:t))
     # gamma2 = []
     #
     # Note: we only compute xi if it is requested as a return argument, since it can be very large.
     # Similarly, we only compute gamma2 on request (and if using MOG outputs).
     #
     # Examples:
     #
     # [alpha, beta, gamma, loglik] = fwdback(pi, A, multinomial_prob(sequence, B));
     #
     # [B, B2] = mixgauss_prob(data, mu, Sigma, mixmat);
     # [alpha, beta, gamma, loglik, xi, gamma2] = fwdback(pi, A, B, 'obslik2', B2, 'mixmat', mixmat);
 
     if isinstance(obslik, tuple):
          obslik = obslik[0]
 
     Q = len(obslik)
     T = len(obslik[0])

     if len(obslik2) == 0:
          compute_gamma2 = 0

     if len(act) == 0: #not storing act info input transmat, but need to add dummy 3rd dim..
          act = np.zeros(T)
          test = np.zeros((1,len(transmat),len(transmat[0])))
          test[0] = transmat
          transmat = test  #transmat[act=0][i][j]
     scale = np.ones(T)

     # scale(t) = Pr(O(t) | O(1:t-1)) = 1/c(t) as defined by Rabiner (1989).
     # Hence prod_t scale(t) = Pr(O(1)) Pr(O(2)|O(1)) Pr(O(3) | O(1:2)) ... = Pr(O(1), ... ,O(T))
     # or log P = sum_t log scale(t).
     # Rabiner suggests multiplying beta(t) by scale(t), but we can instead
     # normalise beta(t) - the constants will cancel when we compute gamma.

     loglik = 0
     alpha = np.zeros((Q,T))
     gamma = np.zeros((Q,T))
     if bool(compute_xi):
          xi_summed = np.zeros((Q,Q))
     else:
          xi_summed = np.array([])
     
     ###########################  Forward ########################################################

     t = 0
    # alpha[:,0] = init_state_distrib[:] * obslik[:,t]
    # print "obslik = ", obslik
    # print "alpha = ",alpha
     alpha[:,0] = init_state_distrib * obslik[:,t]
     if bool(scaled):
          alpha[:,t],scale[t] = normalize(alpha[:,t])
     #assert(approxeq(sum(alpha(:,t)),1))
     for t in range(1,T):
          trans = transmat[int(act[t-1])]
          if bool(maximize):
               m = np.max(np.transpose(trans)*alpha[:,t-1],axis=1)
          else:
               m = np.dot(np.transpose(trans),alpha[:,t-1])
          alpha[:,t] = m*obslik[:,t]
          if bool(scaled):
               alpha[:,t], scale[t] = normalize(alpha[:,t])
          if bool(compute_xi and fwd_only): # useful for online EM
               norm,z = normalize(np.dot(alpha[:,t-1],obslik[:,t]).T*trans)
               xi_summed = xi_summed + norm
     if bool(scaled):
          if len(np.where( scale==0 )[0])>0:
               loglik = -1*np.inf
          else:
               loglik = np.sum(np.log(scale))
     else:
          loglik = np.log(np.sum(alpha[:,T-1]))

     if bool(fwd_only):
          gamma = alpha
          beta = np.array([])
          gamma2 = np.array([])
          return alpha,beta,gamma,loglik,xi_summed,gamma2

     ##################### Backwards #############################
 
     beta = np.zeros((Q,T))
     if bool(compute_gamma2):
          if mixmat.ndim > 2:
               M = len(mixmat[0][0])
          elif mixmat.ndim == 2:
               M = len(mixmat[0])
          else:
               M = 1
          gamma2 = np.zeros((Q,M,T))
     else:
          gamma2 = np.array([])
     
     beta[:,T-1] = np.ones(Q)
     gamma[:,T-1],z = normalize(alpha[:,T-1]*beta[:,T-1])
     t = T-1
     if bool(compute_gamma2):
          values = obslik[:,t]
          values[values==0] = 1
          denom = values
     #     if isinstance(mixmat,list):
          if mixmat.ndim > 2: #corresponds to cell structure in Matlab code
               if len(obslik2) == 0:
                    gamma2 = np.array([])
               else:
                    gamma2[:,:,t] = obslik2[:,:,t] * mixmat[t] * np.matlib.repmat(gamma[:,t],M,1).T/np.matlib.repmat(denom,M,1).T #not applicable action > 1 (POMDP)
          else:
               #prob obs * prob at comp mix * prob at hidden state from alpha forward and beta backwards
               gamma2[:,:,t] = obslik2[:,:,t] * mixmat * np.matlib.repmat(gamma[:,t],M,1).T/np.matlib.repmat(denom,M,1).T 

     for t in range(T-2,-1,-1):
          b = beta[:,t+1]*obslik[:,t+1]
          trans = transmat[int(act[t])]
          if bool(maximize):
               beta[:,t] = np.max(trans*b,axis=1)
          else:
               beta[:,t] = np.dot(trans,b)
          if bool(scaled):
               beta[:,t], z = normalize(beta[:,t])
          gamma[:,t],z = normalize(alpha[:,t]*beta[:,t])
          if bool(compute_xi):
               norm,z = normalize(trans*np.outer(alpha[:,t],b.T))
               xi_summed = xi_summed + norm
          if bool(compute_gamma2):
               values = obslik[:,t]
               values[values==0] = 1
               denom = values
         #      if isinstance(mixmat,list):
               if mixmat.ndim > 2:  #corresponds to cell structure in Matlab code
                    if len(obslik2) == 0:
                         gamma2 = np.array([])
                    else:
                         gamma2[:,:,t] = obslik2[:,:,t] * mixmat[t] * np.matlib.repmat(gamma[:,t],M,1).T/np.matlib.repmat(denom,M,1).T
               else:
                    gamma2[:,:,t] = obslik2[:,:,t] * mixmat * np.matlib.repmat(gamma[:,t],M,1).T/np.matlib.repmat(denom,M,1).T
     return alpha,beta,gamma,loglik,xi_summed,gamma2
# We now explain the equation for gamma2
# Let zt=y(1:t-1,t+1:T) be all observations except y(t)
# gamma2(Q,M,t) = P(Qt,Mt|yt,zt) = P(yt|Qt,Mt,zt) P(Qt,Mt|zt) / P(yt|zt)
#                = P(yt|Qt,Mt) P(Mt|Qt) P(Qt|zt) / P(yt|zt)
# Now gamma(Q,t) = P(Qt|yt,zt) = P(yt|Qt) P(Qt|zt) / P(yt|zt)
# hence
# P(Qt,Mt|yt,zt) = P(yt|Qt,Mt) P(Mt|Qt) [P(Qt|yt,zt) P(yt|zt) / P(yt|Qt)] / P(yt|zt)
#                = P(yt|Qt,Mt) P(Mt|Qt) P(Qt|yt,zt) / P(yt|Qt)


def main():

     prior1 = np.loadtxt("testfiles/priorfb.csv",delimiter = ',')
     transmat1 = np.loadtxt("testfiles/transmatfb.csv",delimiter = ',')
     B = np.loadtxt("testfiles/Bfb.csv",delimiter = ',')

     alphaTest,betaTest,gammaTest,logpTest,xi_summedTest,gamma2Test = fwdback2(prior1,transmat1,B,compute_xi=1)

     #Actual outputs:
     xi_summed =np.loadtxt("testfiles/xi_summedfb.csv",delimiter = ',')
     alpha = np.loadtxt("testfiles/alphafb.csv",delimiter =',')
     beta = np.loadtxt("testfiles/betafb.csv",delimiter = ',')
     logp = np.loadtxt("testfiles/current_loglikfb.csv",delimiter = ',')
     gamma = np.loadtxt("testfiles/gammafb.csv",delimiter = ',')

     print("gamma = ",gammaTest)
     print("alpha = ",alphaTest)
     print("beta = ",betaTest)
     print("logp = ",logpTest)
     print("xi_summed = ", xi_summedTest)

     prior1 = np.loadtxt("testfiles/priormfb.csv",delimiter = ',')
     transmat1 = np.loadtxt("testfiles/transmatmfb.csv",delimiter = ',')
     B = np.loadtxt("testfiles/Bmfb.csv",delimiter = ',')
     B2 = np.loadtxt("testfiles/B2mfb.csv",delimiter = ',')
     mixmat_val = np.loadtxt("testfiles/mixmatmfb.csv",delimiter = ',')
     B2 = np.reshape(B2,(3,2,62707),order = 'F')

     alphaTest,betaTest,gammaTest,logpTest,xi_summedTest,gamma2Test = fwdback2(prior1,transmat1,B,obslik2 = B2,mixmat = mixmat_val,compute_xi=1,compute_gamma2 = 1)
   
     print("gamma = ",gammaTest)
     print("gamma2 shape = ", gamma2Test.shape)
     print("gamma2[:,:,0] = ",gamma2Test[:,:,0])
     print("gamma2[:,:,1] = ",gamma2Test[:,:,1])
     print("gamma2[:,:,2] = ",gamma2Test[:,:,2])
     print("gamma2[:,:,3] = ",gamma2Test[:,:,3])
     print("alpha = ",alphaTest)
     print("beta = ",betaTest)
     print("logp = ",logpTest)
     print("xi_summed = ", xi_summedTest)

 #    alphaTest,betaTest,gammaTest,logpTest,xi_summedTest,gamma2Test = fwdback2(prior1,transmat1,B,obslik2 = B2,mixmat = mixmat_val,compute_xi=1,compute_gamma2 = 1, maximize = 1)

 #    print "gamma = ",gammaTest
 #    print "gamma2 shape = ", gamma2Test.shape
 #    print "gamma2[:,:,0] = ",gamma2Test[:,:,0]
 #    print "gamma2[:,:,1] = ",gamma2Test[:,:,1]
 #    print "gamma2[:,:,2] = ",gamma2Test[:,:,2]
 #    print "gamma2[:,:,3] = ",gamma2Test[:,:,3]
 #    print "alpha = ",alphaTest
 #    print "beta = ",betaTest
 #    print "logp = ",logpTest
 #    print "xi_summed = ", xi_summedTest

if __name__ == "__main__":
     main()

