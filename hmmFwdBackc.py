import ctypes
import os
import numpy as np
import time

#taken from:
#http://pyopengl.sourceforge.net/pydoc/numpy.ctypeslib.html
#http://stackoverflow.com/questions/22425921/pass-a-2d-numpy-array-to-c-using-ctypes/27737099


_mod = ctypes.cdll.LoadLibrary('./_hmmFwdBack.so') 
array_1d_double = np.ctypeslib.ndpointer(dtype=np.double,ndim=1, flags='CONTIGUOUS')
array_2d_double = np.ctypeslib.ndpointer(dtype=np.double,ndim=2, flags='CONTIGUOUS') #I have trouble with array_2d_double,_doublepp instead
_doublepp = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags='C') 


_mexFunction = _mod.mexFunction
_mexFunction.restype = ctypes.c_double
_mexFunction.argtypes = [_doublepp, _doublepp,_doublepp,array_1d_double,_doublepp,_doublepp,ctypes.c_int,ctypes.c_int,ctypes.c_int]

#start_time = time.time()
def MF(initdist,transmat,softev,M):
     start_time = time.time()
  
     out_alpha = np.zeros_like(softev)
     app = (out_alpha.__array_interface__['data'][0]
      + np.arange(out_alpha.shape[0])*out_alpha.strides[0]).astype(np.uintp)
     out_beta = np.zeros_like(softev) 
     bpp = (out_beta.__array_interface__['data'][0]
      + np.arange(out_beta.shape[0])*out_beta.strides[0]).astype(np.uintp)
     out_gamma = np.zeros_like(softev) 
     gpp = (out_gamma.__array_interface__['data'][0]
      + np.arange(out_gamma.shape[0])*out_gamma.strides[0]).astype(np.uintp)
   
     transmatpp = (transmat.__array_interface__['data'][0]
      + np.arange(transmat.shape[0])*transmat.strides[0]).astype(np.uintp)

     SEpp = (softev.__array_interface__['data'][0]
      + np.arange(softev.shape[0])*softev.strides[0]).astype(np.uintp)
     logp = _mexFunction(app,bpp,gpp,initdist,transmatpp,SEpp,len(transmat),len(softev[0]),M)
     print("--- %s seconds ---" % (time.time() - start_time))
     return out_alpha,out_beta,out_gamma,logp

def main():

    prior1 = np.loadtxt("testfiles/prior1.csv",delimiter = ',')
 
    transmat1 = np.loadtxt("testfiles/transmat1.csv",delimiter = ',')

    B = np.loadtxt("testfiles/Bval.csv",delimiter = ',')

    oa,ob,og,logp = MF(prior1,transmat1,B,1)

    print("oa = ",oa)
    print("ob = ",ob)
    print("og = ",og)
    print("logp = ",logp)

if __name__ == "__main__":
     main()

