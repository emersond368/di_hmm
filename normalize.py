import numpy as np

def normalize(A, dim=0):

     #NORMALISE Make the entries of a (multidimensional) array sum to 1
     #[M, c] = normalise(A)
     #c is the normalizing constant
     #
     #[M, c] = normalise(A, dim)
     #If dim is specified, we normalise the specified dimension only,
     #otherwise we normalise the whole array.

     if dim == 0: #normalize vector
         z = np.sum(A)
         # Set any zeros to one before dividing
         if z == 0:
             z = 1
         M = A/z
     elif dim == 1: #normalize column of matrix
         z = np.sum(A, axis = 0)
         z[z==0] = 1
         M = A/z

     return M,z

def main():

     test = np.array([ 0.00389505,0.03523699,0.01176867])
     M,z = normalize(test)
     print(M)

if __name__ == "__main__":
     main()
     
         
