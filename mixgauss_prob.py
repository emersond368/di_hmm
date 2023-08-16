import numpy as np
from numpy.linalg import inv
from sqdist import sqdist
from gaussian_prob import gaussian_prob

def mixgauss_prob(data,mu,Sigma,*mixmatunit_norm):

     # EVAL_PDF_COND_MOG Evaluate the pdf of a conditional mixture of Gaussians
     # function [B, B2] = eval_pdf_cond_mog(data, mu, Sigma, mixmat, unit_norm)
     #
     # Notation: Y is observation, M is mixture component, and both may be conditioned on Q.
     # If Q does not exist, ignore references to Q=j below.
     # Alternatively, you may ignore M if this is a conditional Gaussian.
     #
     # INPUTS:
     # data(:,t) = t'th observation vector 
     #
     # mu(:,k) = E[Y(t) | M(t)=k] 
     # or mu(:,j,k) = E[Y(t) | Q(t)=j, M(t)=k]
     #
     # Sigma(:,:,j,k) = Cov[Y(t) | Q(t)=j, M(t)=k]
     # or there are various faster, special cases:
     #   Sigma() - scalar, spherical covariance independent of M,Q.
     #   Sigma(:,:) diag or full, tied params independent of M,Q. 
     #   Sigma(:,:,j) tied params independent of M. 
     #
     # mixmat(k) = Pr(M(t)=k) = prior
     # or mixmat(j,k) = Pr(M(t)=k | Q(t)=j) 
     # Not needed if M is not defined.
     #
     # unit_norm - optional; if 1, means data(:,i) AND mu(:,i) each have unit norm (slightly faster)
     #
     # OUTPUT:
     # B(t) = Pr(y(t)) 
     # or
     # B(i,t) = Pr(y(t) | Q(t)=i) 
     # B2(i,k,t) = Pr(y(t) | Q(t)=i, M(t)=k) 
     #
     # If the number of mixture components differs depending on Q, just set the trailing
     # entries of mixmat to 0, e.g., 2 components if Q=1, 3 components if Q=2,
     # then set mixmat(1,3)=0. In this case, B2(1,3,:)=1.0.

     mu = np.array(mu,dtype = np.float64)     #casting input to float to safeguard
     data = np.array(data,dtype = np.float64)

     if mu.ndim == 1:  #vector
          d = 1
          Q  = len(mu)
          M = 1
     elif mu.ndim == 2:  #2D matrix
          d = len(mu)
          Q = len(mu[0])
          M = 1
     else:              #3D matrix
          d = len(mu)
          Q = len(mu[0])
          M = len(mu[0][0])
     
     if data.ndim > 1:
          d = len(data)   #yes, this d replaces prior d's which are not needed
          T = len(data[0])
     else:
          d = 1
          T = len(data)

     if not mixmatunit_norm:
         mixmat = np.ones((Q,1))
         unit_norm = 0
     elif len(list(mixmatunit_norm)) == 1:
         mixmat = list(mixmatunit_norm)[0]
         unit_norm = 0
     else:
         mixmat = list(mixmatunit_norm)[0]
         unitnorm = list(mixmatunit_norm)[1]
     
     if np.isscalar(Sigma):
         mu = np.reshape(mu,(d,Q*M))
         if bool(unit_norm):
             print("nonzero unit_norm used: ",unit_norm)
             D = 2 - 2*(np.dot(np.transpose(mu),data))
             D2 = np.transpose(sqdist(data,mu))
             diff = np.abs(D-D2)
             test = np.all(diff < 0.01)  #tolerance (0.01), D ~= D 
             assert(test)
         else:
             D = np.transpose(sqdist(data,mu))

         logB2 = -(d/2)*np.log(2*np.pi*Sigma) - np.dot((1/(2*Sigma)),D)
         B2 = np.reshape(np.exp(logB2),(Q,M,T),order = 'F')

     elif Sigma.ndim == 2:  #tied full
         mu = np.reshape(mu,(d,Q*M))
         D = np.transpose(sqdist(data,mu,inv(Sigma)))
         sign,logdetS = np.linalg.slogdet(Sigma)
         logB2 = -(d/2)*np.log(2*np.pi)-0.5*logdetS - 0.5*D
         B2 = np.reshape(np.exp(logB2),(Q,M,T),order = 'F' )

     elif Sigma.ndim == 3: #tied across M
         B2 = np.zeros((Q,M,T))
         for j in range(0,Q):
             # D[m,t] = sq dist between data[:,t] and mu[:,j,m]
             try:
                  np.linalg.cholesky(Sigma[:,:,j])  #raises LinAlgError if not symmetric,positive definite
             except LinAlgError:
                  print("Sigma not positive definite")
             else:
                  if mu.ndim == 1:
                       invval= inv(Sigma[:,:,j])
                       if invval.shape == (1,1):
                            invval = invval[0][0]
                       elif len(invval) == 1:
                            invval = invval[0]
                       D = sqdist(data,mu[j],invval).T
                  else:
                       invval= inv(Sigma[:,:,j])
                       if invval.shape == (1,1):
                            invval = invval[0][0]
                       elif len(invval) == 1:
                            invval = invval[0]
                       D = sqdist(data,mu[:,j],invval).T
                  sign,logdetS = np.linalg.slogdet(Sigma[:,:,j])
                  logB2 = -(float(d)/2)*np.log(2*np.pi) - 0.5*logdetS - 0.5*D
                  B2[j,:,:] = np.exp(logB2)
     else: #general case
          B2 = np.zeros((Q,M,T))
          for j in range(0,Q):
               for k in range(0,M):
                    B2[j,k,:] = gaussian_prob(data,mu[:,j,k],Sigma[:,:,j,k])
           
     B = np.zeros((Q,T))
     if Q < T:
          for q in range(0,Q):
               if M==1:
                     B[q,:] = np.dot(mixmat[q],B2[q,:,:]) #vector * matrix sums over m 
               else:
                     B[q,:] = np.dot(mixmat[q,:],B2[q,:,:]) #vector * matrix sums over m
     else:
          for t in range(0,T):
               B[:,t] = np.sum(mixmat*B2[:,:,t],axis = 1) # sum over m
         
     return B,B2
             
def main():
   
     mu = np.loadtxt("testfiles/mumgp.csv",delimiter = ',')
     Sigma = np.loadtxt("testfiles/Sigmamgp.csv", delimiter = ',')
     obs = np.loadtxt("testfiles/obsmgp.csv", delimiter = ',')
     Sigma = np.reshape(Sigma,(1,1,3))
     #Actual result:
     B = np.loadtxt("testfiles/Bmgp.csv",delimiter = ',')
     
     Btest,B2test = mixgauss_prob(obs,mu,Sigma)

     print("Btest.shape = ", Btest.shape)
     print("Btest = ", Btest)
     print("B2test = ", B2test)

     mu = np.loadtxt("testfiles/mummgp.csv",delimiter = ',')
     mu = np.zeros((1,3,2))
     mu[:,:,0] = np.array([-0.7278,-1.7447,-0.0626])
     mu[:,:,1] = np.array([1.0743,0.0068,-0.5040])
     Sigma = np.loadtxt("testfiles/Sigmammgp.csv", delimiter = ',')
     Sigma = np.zeros((1,1,3,2))
     Sigma[0,0,0,0] = 3.7166e+04
     Sigma[0,0,1,0] = 1.1112e+06
     Sigma[0,0,2,0] = 9.0864
     Sigma[0,0,0,1] = 3.1369e+03
     Sigma[0,0,1,1] = 0.0530
     Sigma[0,0,2,1] = 222.0210
     obs = np.loadtxt("testfiles/obsmmgp.csv", delimiter = ',')
     mixmat = np.loadtxt("testfiles/mixmatmmgp.csv", delimiter = ',')

     Btest,B2test = mixgauss_prob(obs,mu,Sigma,mixmat)

     print("Btest.shape = ", Btest.shape)
     print("Btest = ", Btest)
     print("B2test = ", B2test)

     testX = np.loadtxt("testfiles/testXmmhmm.csv",delimiter = ',')
     mixmat1 = np.loadtxt("testfiles/mixmat1mmhmm.csv",delimiter = ',')
     Sigma1 = np.loadtxt("testfiles/Sigma1mmhmm.csv",delimiter = ',')
     mu1 = np.loadtxt("testfiles/mu1mmhmm.csv",delimiter = ',')
     Sigma1 = np.reshape(Sigma1,(1,1,3,2), order = 'F')
     mu1 = np.reshape(mu1,(1,3,2), order = 'F')

     B,B2 = mixgauss_prob(testX,mu1,Sigma1,mixmat1)
     print("B = ",B)

if __name__ == "__main__":
     main()

         


