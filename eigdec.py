import numpy as np
from scipy.sparse.linalg import eigs


def eigdec(x,N):
     #EIGDEC	Sorted eigendecomposition
     #
     #	Description
     #	 EVALS = EIGDEC(X, N computes the largest N eigenvalues of the
     #	matrix X in descending order.  [EVALS, EVEC] = EIGDEC(X, N) also
     #	computes the corresponding eigenvectors.
     #
     #	See also
     #	PCA, PPCA
     #

     #	Copyright (c) Ian T Nabney (1996-2001)

     x = np.array(x, dtype = float)

  #   if nargout == 1:
  #       evals_only = True
  #   else:
  #       evals_only = False

     try:
          if ((N != np.round(N)) or (N < 1) or (N > len(x[0]))):
                raise ValueError
     except ValueError:
          print "Number of PCs must be integer, >0, < dim"

     # Find the eigenvalues of the data covariance matrix
  #   if evals_only:
  #        temp_evals, temp_evecs = np.linalg.eig(x)
  #   else:
          #Use eig function unless fraction of eigenvalues required is tiny
     if (float(N)/float(len(x[0]))) > 0.04:
           print "netlab pca: using eig"
           temp_evals, temp_evecs = np.linalg.eig(x)
           temp_evecs = -1*temp_evecs  # technically does not matter, matlab eigv was neg of this
  #         print "temp_evals = ",temp_evals
  #         print "temp_evecs = ",temp_evecs
     else:
  #         options.display = 0
            print "netlab pca: using eigs"
            temp_evals, temp_evecs = eigs(x, N, which='LM')

     # Eigenvalues nearly always returned in descending order, but just to ensure...
     
     #if isinstance(temp_evals[0],complex):
     #     evals = np.sort(np.abs(temp_evals),kind='mergesort')
     #     perm = np.zeros(len(evals))
     #     for i in range(0,len(evals)):
     #          perm[i] = np.where(evals[i] == np.abs(temp_evals))[0][0] #collect row value
     #     evecs = np.array(temp_evecs)
     #     fevals = np.array(temp_evals)
     #     if bool((evals == np.abs(temp_evals)).all()):
     #          evecs = temp_evecs[:,0:N]
     #          fevals = np.array(temp_evals[0:N])
     #     else:
     #          print "pca: sorting evec"
     #          for i in range(0,N):
     #               evecs[:,i] = temp_evecs[:,perm[i]]
     #               fevals[i] = np.array(temp_evals[perm[i]])
     #else:
     evals = np.sort(-1*temp_evals,kind='mergesort')  #from largest to smallest
     perm = np.zeros(len(evals))
     for i in range(0,len(evals)):
           perm[i] = np.where(evals[i] == -1*(temp_evals))[0][0] #collect row value
     evecs = np.array(temp_evecs)
     fevals = -evals
     if bool((fevals == temp_evals).all()):
          evecs = temp_evecs[:,0:N]
          fevals = np.array(temp_evals[0:N])
     else:
          print "pca: sorting evec"
          for i in range(0,N):
               evecs[:,i] = temp_evecs[:,perm[i]]
               fevals[i] = np.array(temp_evals[perm[i]])
     if len(temp_evecs[0]) > N:
    #      np.delete(evecs,np.s_[N:len(evecs[0])],axis = 0)
           evecs = evecs[0:N,0:N]
           fevals = fevals[0:N]
     return fevals,evecs

def main():

     H = np.array([[1,2,3,-4,-6,7,0],[2,4,5,6,6.6,7,9],[8,0,-7,-2,2,7,11],[9,-4.5,1.3,4,0,7,12],[-4, 0,3.2,1,-9,6,0],[0,0,-4,7,9,5,5],[9,1.1,3,4,5,0,-7]])

  #   H = (H + H.T)/2

     evals,evecs = eigdec(H,7)  

     print evals
     print evecs

     evals,evecs = eigdec(H,5)

     print evals
     print evecs

     H = np.array([[0.40384,0.08292,-0.00195,0.09058],[0.08292,0.54075,0.11852,0.10496],[-0.00195,0.11852,0.76583,-0.09785],[0.09058,0.10496,-0.09785,0.43165]])
     data_dim = 4
     evals,evecs = eigdec(H,data_dim)

     print evals
     print evecs


if __name__ == "__main__":
     main()
            
