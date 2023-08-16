from eigdec import eigdec
import numpy as np
import warnings

def ppca(x, ppca_dim):

     #PPCA	Probabilistic Principal Components Analysis
     #
     #	Description
     #	 [VAR, U, LAMBDA] = PPCA(X, PPCA_DIM) computes the principal
     #	component subspace U of dimension PPCA_DIM using a centred covariance
     #	matrix X. The variable VAR contains the off-subspace variance (which
     #	is assumed to be spherical), while the vector LAMBDA contains the
     #	variances of each of the principal components.  This is computed
     #	using the eigenvalue and eigenvector  decomposition of X.
     #
     #	See also
     #	EIGDEC, PCA
     #

     #	Copyright (c) Ian T Nabney (1996-2001)
     
     x = np.array(x, dtype = float)
     #print "x = ", x
     #print "ppca_dim = ",ppca_dim


     try:
          if ((ppca_dim != np.round(ppca_dim)) or (ppca_dim < 1) or (ppca_dim > len(x[0]))):
               raise ValueError
     except ValueError:
          print "Number of PCs must be integer, >0, < dim"

     
     ndata,data_dim = x.shape
  
     # Assumes that x is centred and responsibility weighted covariance matrix
     l,Utemp = eigdec(x,data_dim)
     # Zero any negative eigenvalues (caused by rounding)
     l[l<0] = 0

     # Now compute the sigma squared values for all possible values of q
     s2_temp = np.cumsum(l[::-1])/range(1,data_dim+1)
     #If necessary, reduce the value of q so that var is at least
     # eps * largest eigenvalue
     
     
     if l[0] == 0:
          inter = np.zeros(len(s2_temp))
          for i in range(0,len(s2_temp)):
               if s2_temp[i] != 0:  #value is inf and therefore treated as above eps otherwise not
                    inter[i] = 1
          q_temp =  np.min([ppca_dim,data_dim - (np.min(np.where(inter > np.finfo(float).eps)[0])+1)])
     else:
          #print "np.where(s2_temp/l[0] > np.finfo(float).eps) = ",np.where(s2_temp/l[0] > np.finfo(float).eps)
          q_temp =  np.min([ppca_dim,data_dim - (np.min(np.where(s2_temp/l[0] > np.finfo(float).eps)[0])+1)])  #add one - index at 0
     if q_temp != ppca_dim:
          warnings.warn("Covariance matrix ill-conditioned: extracted")

     if q_temp == 0:
          #All the latent dimensions have disappeared, so we are just left with the noise model
          var = l[0]/data_dim;
          lam = var*np.ones(ppca_dim)
          lam[0:q_temp] = l[0:q_temp]
     else:
          var = np.mean(l[q_temp:len(l)])
          lam = l[0:q_temp]
     U = Utemp[:,0:q_temp]

     return var,U,lam    

def main():

     H = [[1,2,3,-4,-6,7,0],[2,4,5,6,6.6,7,9],[8,0,-7,-2,2,7,11],[9,-4.5,1.3,4,0,7,12],[-4, 0,3.2,1,-9,6,0],[0,0,-4,7,9,5,5],[9,1.1,3,4,5,0,-7]]

     var,U,lam = ppca(H,5)

     print "var = ",var
     print "U = ",U
     print "lam = ",lam 

     #Results SHOULD BE:
     #lam = np.array([0.0000 + 0.0000j,  6.1276 + 0.0000j, 1.7644 + 8.5353j,  1.7644 - 8.5353j]) 
     #U = np.array([[0.1392 + 0.0000j, -0.2442 + 0.0000j,-0.1739 + 0.3878j,-0.1739 - 0.3878j],\
     #               [0.5358 + 0.0000j,-0.8689 + 0.0000j,0.0381 - 0.2466j,0.0381 + 0.2466j],\
     #               [-0.5254 + 0.0000j,-0.2213 + 0.0000j,-0.0024 + 0.3059j,-0.0024 - 0.3059j],\
     #               [0.4503 + 0.0000j,-0.0402 + 0.0000j,0.5871 + 0.0000j,0.5871 + 0.0000j],\
     #               [-0.4424 + 0.0000j,0.1077 + 0.0000j,-0.1448 - 0.1279j,-0.1448 + 0.1279j],\ 
     #               [-0.1326 + 0.0000j,0.2336 + 0.0000j,-0.2903 - 0.3400j,-0.2903 + 0.3400j],\
     #               [-0.0379 + 0.0000j,-0.2620 + 0.0000j,0.2049 + 0.1994j,0.2049 - 0.1994j]])
     #var = 5.0455

if __name__ == "__main__":
     main()
