import numpy as np
import numpy.matlib

def mk_stochastic(T):

     # MK_STOCHASTIC Ensure the argument is a stochastic matrix, i.e., the sum over the last dimension is 1.
     # [T,Z] = mk_stochastic(T)
     #
     # If T is a vector, it will sum to 1.
     # If T is a matrix, each row will sum to 1.
     # If T is a 3D array, then sum_k T[i,j,k] = 1 for all i,j.

     # Set zeros to 1 before dividing

     W = np.array(T,dtype = np.float64)
     if (W.ndim == 1): # is a vector
         z = np.sum(W)
         if z == 0:
             z = 1
         W =  W/z
     elif W.ndim == 2: # is a matrix
         z = np.sum(W,axis = 1)  #sum row wise
         for i in range(0,len(z)):
              if z[i] == 0:
                   z[i] = 1
         z.shape = 1,len(z) # ensure single column
         norm = np.matlib.repmat(z,len(W[0]),1)
         norm = np.transpose(norm)
         norm = np.array(norm, dtype = np.float64)
         W = W/norm
     else: #multidimensional array
         ns = np.array([len(W),len(W[0]),len(W[0][0])])  #retreive dim sizes for row,col, height
         W = np.reshape( W,(np.prod(ns[0:-1]),ns[-1])) 
         z = np.sum(W,axis = 1) #sum row wise
         for i in range(0,len(z)):
              if z[i] == 0:
                  z[i] = 1
         norm = np.matlib.repmat(z,len(W[0]),1)
         norm = np.transpose(norm)
         norm = np.array(norm, dtype = np.float64)
         W = W/norm
         W = np.reshape(W, ns)
     return W

def main():
    
    A = np.array([[[5, 7, 8], [0, 0, 0],[4, 3, 6]],[[1, 0, 4],[3, 5, 6],[9, 8, 7]]])
    A = mk_stochastic(A)

    print(A)

if __name__ == "__main__":
     main()
   
