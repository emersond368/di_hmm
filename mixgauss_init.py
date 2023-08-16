import numpy as np
import numpy.matlib

def mixgauss_init(M,data,cov_type,method = 'kmeans'):

     # MIXGAUSS_INIT Initial parameter estimates for a mixture of Gaussians
     # function [mu, Sigma, weights] = mixgauss_init(M, data, cov_type. method)
     #
     # INPUTS:
     # data[:,t] is the t'th example
     # M = num. mixture components
     # cov_type = 'full', 'diag' or 'spherical'
     # method = 'rnd' (choose centers randomly from data) or 'kmeans' (needs netlab)
     #
     # OUTPUTS:
     # mu[:,k] 
     # Sigma(:,:,k) 
     # weights(k)

     data = np.array(data,dtype = np.float)
     (d,T) = size(data)

     data = np.reshape(data,(d,T)) #in case it is data(:, t, sequence_num)

     if method == 'rnd':
          C = np.cov(data.T,rowvar=False)
          Sigma = np.zeros((len(C),len(C),M))
          for i in range(0,M):
               Sigma[:,:,i] = np.diag(np.diag(C))*0.5
          indices = np.random.permutation(T)
          mu = data[:,indices[0:M]]
          weights = normalize(np.ones((M,1)))
     else:
          mix = gmm(d,M,cov_type)
          foptions = np.array([0,0.0001,0.0001,0.000001,0,0,0,0,0,0,0,0,0,0,0,0.00000001,0.1,0])
          options = foptions 
          max_iter = 5
          options[0] = -1 #Be quiet!
          options[13] = max_iter
          mix = gmminit(mix,data.T,options)
          mu = np.reshape(mix["centres"].T,(d,M),order = 'F')
          weights = mix["priors"][:]
          for j in range(0,M):
               if cov_type == 'diag':
                   Sigma[:,:,j] = np.diag(mix["covars"][j,:])
               elif cov_type == 'full':
                   Sigma[:,:,j] = mix["covars"][:,:,j]
               elif cov_type == 'spherical'
                   Sigma[:,:,j] = np.dot(mix["covars"][j],np.eye(d))

def normalize(col):
    z = np.sum(col)
    if z == 0:
         z = 1
    return(col/z)


