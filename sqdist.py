import numpy as np
import numpy.matlib
import unittest

def sqdist(p,q,*A):

     # SQDIST      Squared Euclidean or Mahalanobis distance.
     # SQDIST(p,q)   returns m(i,j) = (p(:,i) - q(:,j))'*(p(:,i) - q(:,j)).
     # SQDIST(p,q,A) returns m(i,j) = (p(:,i) - q(:,j))'*A*(p(:,i) - q(:,j)).

     p = np.array(p,dtype = np.float)
     q = np.array(q,dtype = np.float)
     A = np.array(A,dtype = np.float)  #added extra dimension because optional arg forms a list of added parameters
    
     if p.ndim == 0:
          pn = 1 
     elif p.ndim == 1:
          pn = len(p)
     else:
          pn = len(p[0])
     if q.ndim == 0:
          qn = 1
     elif q.ndim == 1:
          qn = len(q)
     else:
          qn = len(q[0])

     if (len(A) == 0):
          pmag = np.sum(p*p,axis=0) # sum column wise
          qmag = np.sum(q*q,axis=0) 
          m = np.matlib.repmat(qmag,pn,1) + np.transpose(np.matlib.repmat(pmag,qn,1)) - 2*np.dot(np.transpose(p),q)
     else:
          errtest = np.array(A,dtype = np.float)
          if (errtest.size == 0) or (p.size == 0):
               print("sqdist: empty matrices Error")
               assert 1==0
          A = A[0]
          if ((A.ndim == 1) and len(A) == 1): #A is a scalar
               Ap = A*p
               Aq = A*q
          else:
               Ap = np.dot(p,A)
               Aq = np.dot(A,q)
          if (p.ndim == 1):
               pmag = p*Ap
          else:
               pmag = np.sum(p*Ap,axis = 0)
          if (q.ndim == 1):
               qmag = q*Aq
          else:
               qmag = np.sum(q*Aq,axis = 0)
          if ((q.ndim == 0) or (p.ndim == 0)):
               midval = 2*np.transpose(p)*Aq
               t1 = qmag*np.ones(pn)
               t2 = pmag*np.ones(qn)

               m = t1 + t2 - midval
          else:
               m = np.matlib.repmat(qmag,pn,1) + np.transpose(np.matlib.repmat(pmag,qn,1)) - 2*np.dot(np.transpose(p),Aq)
     return m   

def main():
 
    p = np.array([[1,2,3],[4,5,6],[1.2,4,0]])
    q = np.array([[3,2,5],[4.1,4,5],[1,0,0]])
    A = np.array([[3.6,0,4],[1,0,0],[0,0,0]])

    test = sqdist(p,q)
    print(test)

    test = sqdist(p,q,A)
    print(test)

    A = 5
    test = sqdist(p,q,A)
    print(test)

if __name__ == "__main__":
     main()
  

