import numpy as np

def dist2(x,c):
    
     #DIST2	Calculates squared distance between two sets of points.
     #
     #	Description
     #	D = DIST2(X, C) takes two matrices of vectors and calculates the
     #	squared Euclidean distance between them.  Both matrices must be of
     #	the same column dimension.  If X has M rows and N columns, and C has
     #	L rows and N columns, then the result has M rows and L columns.  The
     #	I, Jth entry is the  squared distance from the Ith row of X to the
     #	Jth row of C.
     #
     #	See also
     #	GMMACTIV, KMEANS, RBFFWD
     #

     #	Copyright (c) Ian T Nabney (1996-2001)
     x = np.array(x, dtype = float)
     c = np.array(c, dtype = float) 

     (ndata,dimx) = x.shape
     (ncentres,dimc) = c.shape

     try:
         if int(dimx) == int(dimc):
               print dimx, dimc
         else:
               raise ValueError
     except ValueError:
               print "Data dimension does not match dimension of centres"

     n2 = np.outer(np.ones(ncentres),np.sum(x*x,axis = 1).T).T + np.outer(np.ones(ndata),np.sum(c*c,axis=1))-2*np.dot(x,c.T)

     # Rounding errors occasionally cause negative entries in n2 
     n2[n2 < 0] = 0

     return n2

     
